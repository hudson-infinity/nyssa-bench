'use strict';

const {conventionalTitle, needsContainers} = require('./policy.cjs');
const {readMergeState} = require('./api.cjs');

async function pullFiles(github, repo, pr) {
  if (pr.changed_files > 3000) throw new Error('PR exceeds the GitHub file-list limit; refusing an incomplete CI decision.');
  const files = await github.paginate(github.rest.pulls.listFiles, {
    ...repo, pull_number: pr.number, per_page: 100,
  });
  return files.flatMap(file => [file.filename, file.previous_filename].filter(Boolean));
}

async function ciPolicy({github, context, core}) {
  let pr = context.payload.pull_request;
  if (!pr && context.ref !== 'refs/heads/main') {
    const pulls = await github.paginate(github.rest.pulls.list, {
      ...context.repo, state: 'open', base: 'main',
      head: `${context.repo.owner}:${context.ref.replace('refs/heads/', '')}`, per_page: 100,
    });
    pr = pulls.find(item => item.head.sha === context.sha);
  }
  if (pr) {
    pr = (await github.rest.pulls.get({...context.repo, pull_number: pr.number})).data;
    conventionalTitle(pr.title, pr.body || '');
    // Exercise the merge bot's actual read query with the CI token as well.
    await readMergeState(github, context.repo, pr.number);
    const files = await pullFiles(github, context.repo, pr);
    core.setOutput('containers', String(needsContainers(files)));
    core.info(`Validated PR #${pr.number}; container CI required: ${needsContainers(files)}`);
  } else {
    // A branch-only preflight checks everything; main has already passed the PR gate.
    core.setOutput('containers', String(context.ref !== 'refs/heads/main'));
  }
}

module.exports = {ciPolicy, pullFiles};
