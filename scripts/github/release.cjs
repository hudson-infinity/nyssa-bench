'use strict';

async function verifyMain({github, context, core}) {
  const branch = (await github.rest.repos.getBranch({...context.repo, branch: 'main'})).data;
  const sha = branch.commit.sha;
  if (context.payload.workflow_run && context.payload.workflow_run.head_sha !== sha) {
    core.setOutput('ready', 'false');
    core.info('Ignoring CI for an older main commit.');
    return;
  }
  const runs = (await github.rest.actions.listWorkflowRuns({
    ...context.repo, workflow_id: 'ci.yml', branch: 'main', head_sha: sha,
    status: 'success', per_page: 100,
  })).data.workflow_runs;
  if (!runs.some(run => run.head_sha === sha && run.conclusion === 'success'
      && ['push', 'workflow_dispatch'].includes(run.event))) {
    throw new Error('Current main must pass CI before version automation can run.');
  }
  core.setOutput('ready', 'true');
}

async function dispatch({github, context, core}) {
  const prs = JSON.parse(process.env.RELEASE_PRS || '[]');
  for (const item of prs) {
    const pr = (await github.rest.pulls.get({...context.repo, pull_number: item.number})).data;
    if (pr.state !== 'open' || pr.base.ref !== 'main'
        || pr.head.repo.full_name !== `${context.repo.owner}/${context.repo.repo}`
        || !pr.head.ref.startsWith('release-please--')) {
      throw new Error('Refusing to dispatch CI for an unexpected release PR.');
    }
    await github.rest.actions.createWorkflowDispatch({
      ...context.repo, workflow_id: 'ci.yml', ref: pr.head.ref,
    });
    await github.rest.actions.createWorkflowDispatch({
      ...context.repo, workflow_id: 'pr-automation.yml', ref: 'main',
      inputs: {pr_number: String(pr.number)},
    });
    core.info(`Dispatched CI and labels for release PR #${pr.number}.`);
  }
  if (process.env.RELEASE_CREATED === 'true') {
    const tag = process.env.RELEASE_TAG;
    if (!/^v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:rc\d+)?$/.test(tag || '')) {
      throw new Error('Release Please returned an invalid release tag.');
    }
    await github.rest.actions.createWorkflowDispatch({
      ...context.repo, workflow_id: 'release.yml', ref: tag,
    });
    core.info(`Dispatched existing publication workflow for ${tag}.`);
  }
}

module.exports = {verifyMain, dispatch};
