'use strict';

const catalog = require('../../.github/labels.json');
const {conventionalTitle, managedLabels, dependencyAutomerge, mergeEligible, checksPass, needsContainers} = require('./policy.cjs');
const {pullFiles} = require('./ci.cjs');
const {readMergeState} = require('./api.cjs');

async function syncCatalog(github, repo) {
  const existing = new Map((await github.paginate(github.rest.issues.listLabelsForRepo,
    {...repo, per_page: 100})).map(label => [label.name, label]));
  for (const label of catalog) {
    const old = existing.get(label.name);
    if (!old) await github.rest.issues.createLabel({...repo, ...label});
    else if (old.color !== label.color || old.description !== label.description) {
      await github.rest.issues.updateLabel({...repo, ...label});
    }
  }
}

async function reconcileLabels(github, repo, pr, files, metadata) {
  const desired = new Set(managedLabels(pr, files));
  const current = new Set(pr.labels.map(label => label.name));
  const owned = catalog.map(label => label.name).filter(name => /^(area:|size:|release:|bot:)/.test(name));
  if (metadata) {
    owned.push('automerge:dependencies');
    const eligible = dependencyAutomerge(pr, metadata.updateType, metadata.maintainerChanges);
    if (eligible) {
      desired.add('automerge:dependencies');
    }
    await github.rest.repos.createCommitStatus({...repo, sha: pr.head.sha,
      context: 'Dependabot policy', state: 'success',
      description: eligible ? 'Verified patch/minor update' : 'Manual opt-in required'});
  }
  for (const name of owned) {
    if (current.has(name) && !desired.has(name)) {
      await github.rest.issues.removeLabel({...repo, issue_number: pr.number, name});
    }
  }
  const additions = [...desired].filter(name => !current.has(name));
  if (additions.length) await github.rest.issues.addLabels({...repo, issue_number: pr.number, labels: additions});
}

async function maybeMerge({github, context, core}, number) {
  const repo = context.repo;
  const pr = (await github.rest.pulls.get({...repo, pull_number: number})).data;
  if (!mergeEligible(pr)) return false;
  if (!pr.labels.some(label => label.name === 'automerge')) {
    const status = (await github.rest.repos.getCombinedStatusForRef({...repo, ref: pr.head.sha})).data;
    if (!status.statuses.some(check => check.context === 'Dependabot policy'
        && check.state === 'success' && check.description === 'Verified patch/minor update'
        && check.creator?.login === 'github-actions[bot]')) return false;
  }
  const files = await pullFiles(github, repo, pr);
  const state = await readMergeState(github, repo, number);
  const contexts = state.commits.nodes[0]?.commit.statusCheckRollup?.contexts;
  if (state.headRefOid !== pr.head.sha || state.mergeable !== 'MERGEABLE'
      || state.mergeStateStatus !== 'CLEAN' || state.reviewDecision === 'CHANGES_REQUESTED'
      || !contexts || contexts.pageInfo.hasNextPage
      || !checksPass(contexts.nodes, needsContainers(files))) {
    core.info(`PR #${number} is waiting for successful checks, current base, or resolved reviews.`);
    return false;
  }
  // Refresh labels and draft state immediately before the SHA-conditional merge.
  const latest = (await github.rest.pulls.get({...repo, pull_number: number})).data;
  if (latest.head.sha !== pr.head.sha || !mergeEligible(latest)) return false;
  conventionalTitle(latest.title, latest.body || '');
  const merged = await github.rest.pulls.merge({
    ...repo, pull_number: number, sha: pr.head.sha, merge_method: 'squash',
    commit_title: `${latest.title} (#${number})`, commit_message: latest.body || '',
  });
  if (!merged.data.merged) throw new Error(`Merge refused: ${merged.data.message}`);
  // GITHUB_TOKEN merges do not emit a usable push trigger for downstream CI.
  await github.rest.actions.createWorkflowDispatch({...repo, workflow_id: 'ci.yml', ref: 'main'});
  core.info(`Squash merged PR #${number}; dispatched main CI.`);
  return true;
}

async function run(args) {
  const {github, context, core} = args;
  await syncCatalog(github, context.repo);
  let numbers = [];
  if (context.payload.pull_request) numbers = [context.payload.pull_request.number];
  else if (context.payload.workflow_run) {
    const run = context.payload.workflow_run;
    if (run.conclusion !== 'success') return;
    const pulls = await github.paginate(github.rest.repos.listPullRequestsAssociatedWithCommit,
      {...context.repo, commit_sha: run.head_sha, per_page: 100});
    numbers = pulls.filter(pr => pr.state === 'open' && pr.head.sha === run.head_sha).map(pr => pr.number);
  } else if (Number(context.payload.inputs?.pr_number) > 0) {
    numbers = [Number(context.payload.inputs.pr_number)];
  } else if (context.eventName === 'schedule') {
    const pulls = await github.paginate(github.rest.pulls.list,
      {...context.repo, state: 'open', base: 'main', per_page: 100});
    numbers = pulls.filter(mergeEligible).map(pr => pr.number);
  }
  for (const number of numbers) {
    const pr = (await github.rest.pulls.get({...context.repo, pull_number: number})).data;
    if (pr.state !== 'open' || pr.base.ref !== 'main') continue;
    const metadata = context.payload.pull_request?.user.login === 'dependabot[bot]'
      ? {updateType: process.env.UPDATE_TYPE, maintainerChanges: process.env.MAINTAINER_CHANGES} : null;
    await reconcileLabels(github, context.repo, pr, await pullFiles(github, context.repo, pr), metadata);
    await maybeMerge(args, number);
  }
  core.info(`Processed ${numbers.length} pull request(s).`);
}

module.exports = {run, maybeMerge, syncCatalog, reconcileLabels};
