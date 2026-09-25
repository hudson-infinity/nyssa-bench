'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const policy = require('./policy.cjs');
const {maybeMerge, reconcileLabels} = require('./pr.cjs');
const {verifyMain, dispatch} = require('./release.cjs');
const {ciPolicy} = require('./ci.cjs');

const pull = () => ({number: 7, state: 'open', draft: false, title: 'fix: repair export', body: '',
  user: {login: 'contributor'}, base: {ref: 'main'}, head: {sha: 'abc'},
  labels: [{name: 'automerge'}], additions: 10, deletions: 2});
const passingChecks = () => policy.REQUIRED_CHECKS.map(name => ({name,
  status: 'COMPLETED', conclusion: 'SUCCESS', checkSuite: {app: {slug: 'github-actions'}}}));
const core = () => ({info() {}, setOutput() {}});

test('conventional titles select the declared version bump', () => {
  for (const [title, bump] of [['fix: repair a bug', 'patch'], ['perf(io): read once', 'patch'],
    ['feat: support a new adapter', 'minor'], ['refactor!: remove an API', 'major'],
    ['chore(deps): bump actions', 'none'], ['docs: clarify exports', 'none']]) {
    assert.equal(policy.conventionalTitle(title).bump, bump);
  }
  assert.equal(policy.conventionalTitle('fix: remove old API', 'BREAKING CHANGE: removed').bump, 'major');
  for (const title of ['Improve things', 'fix: ', 'fix: ok\nfeat!: injected', 'fix(scope\n): bad']) {
    assert.throws(() => policy.conventionalTitle(title));
  }
});

test('container changes cannot bypass the aggregate gate', () => {
  for (const path of ['docker/Dockerfile', 'pyproject.toml', 'uv.lock',
    '.github/workflows/ci.yml', '.github/workflows/release.yml', 'scripts/github/policy.cjs']) {
    assert.equal(policy.needsContainers([path]), true);
  }
  assert.equal(policy.needsContainers(['docs/metrics.md']), false);
  const results = Object.fromEntries(['policy', 'test', 'package', 'installed-artifact', 'automation', 'containers']
    .map(name => [name, {result: 'success'}]));
  assert.doesNotThrow(() => policy.assertGate(results, true));
  for (const result of ['failure', 'cancelled', 'skipped', undefined]) {
    assert.throws(() => policy.assertGate({...results, test: {result}}, true));
  }
  assert.throws(() => policy.assertGate({...results, containers: {result: 'skipped'}}, true));
  assert.doesNotThrow(() => policy.assertGate({...results, containers: {result: 'skipped'}}, false));
});

test('dependency auto-merge requires verified bot metadata and excludes major updates', () => {
  const pr = pull();
  assert.equal(policy.dependencyAutomerge(pr, 'version-update:semver-patch', 'false'), false);
  pr.user.login = 'dependabot[bot]';
  assert.equal(policy.dependencyAutomerge(pr, 'version-update:semver-patch', 'false'), true);
  assert.equal(policy.dependencyAutomerge(pr, 'version-update:semver-minor', 'false'), true);
  assert.equal(policy.dependencyAutomerge(pr, 'version-update:semver-major', 'false'), false);
  assert.equal(policy.dependencyAutomerge(pr, 'version-update:semver-patch', 'true'), false);
  assert.equal(policy.dependencyAutomerge(pr, '', ''), false);
});

test('manual opt-in, drafts, holds, and bot-label spoofing are handled', () => {
  assert.equal(policy.mergeEligible(pull()), true);
  assert.equal(policy.mergeEligible({...pull(), draft: true}), false);
  assert.equal(policy.mergeEligible({...pull(), labels: [{name: 'automerge'}, {name: 'hold'}]}), false);
  assert.equal(policy.mergeEligible({...pull(), labels: []}), false);
  assert.equal(policy.mergeEligible({...pull(), labels: [{name: 'automerge:dependencies'}]}), false);
});

test('all checks must finish successfully and required checks must come from Actions', () => {
  const checks = passingChecks();
  assert.equal(policy.checksPass(checks, false), true);
  assert.equal(policy.checksPass(checks.slice(1), false), false);
  assert.equal(policy.checksPass(checks.map(check => ({...check, checkSuite: {app: {slug: 'other'}}})), false), false);
  for (const conclusion of ['FAILURE', 'CANCELLED', 'TIMED_OUT', 'NEUTRAL', 'SKIPPED', null]) {
    assert.equal(policy.checksPass([...checks, {name: 'additional CI', status: 'COMPLETED', conclusion}], false), false);
  }
  assert.equal(policy.checksPass([...checks, {name: 'additional CI', status: 'QUEUED'}], false), false);
  assert.equal(policy.checksPass([...checks, {context: 'external/status', state: 'PENDING'}], false), false);
  assert.equal(policy.checksPass([...checks, {context: 'PR automation', state: 'FAILURE'}], false), false);
  const skipped = {name: 'containers', status: 'COMPLETED', conclusion: 'SKIPPED'};
  assert.equal(policy.checksPass([...checks, skipped], false), true);
  assert.equal(policy.checksPass([...checks, skipped], true), false);
});

function mockMerge(overrides = {}) {
  const calls = [];
  const pr = pull();
  const state = {headRefOid: pr.head.sha, mergeStateStatus: 'CLEAN', mergeable: 'MERGEABLE',
    reviewDecision: null, commits: {nodes: [{commit: {statusCheckRollup: {contexts: {
      nodes: passingChecks(), pageInfo: {hasNextPage: false},
    }}}}]}, ...overrides};
  const github = {paginate: async () => [{filename: 'docs/example.md'}],
    graphql: async () => ({repository: {pullRequest: state}}),
    rest: {pulls: {get: async () => ({data: pr}), listFiles() {},
      merge: async args => { calls.push(['merge', args]); return {data: {merged: true}}; }},
    actions: {createWorkflowDispatch: async args => calls.push(['dispatch', args])}}};
  return {github, context: {repo: {owner: 'owner', repo: 'repo'}}, core: core(), calls, pr, state};
}

test('superseded workflow runs cannot satisfy or permanently block current checks', () => {
  const inRun = (checks, runNumber, workflow = 'CI') => checks.map((check, index) => ({
    ...check, databaseId: runNumber * 100 + index,
    checkSuite: {app: {slug: 'github-actions'}, workflowRun: {runNumber, workflow: {id: workflow}}},
  }));
  const old = inRun([
    ...passingChecks().map(check => ({...check, conclusion: 'CANCELLED'})),
    {name: 'installed-artifact', status: 'COMPLETED', conclusion: 'CANCELLED'},
    {name: 'containers', status: 'COMPLETED', conclusion: 'SKIPPED'},
  ], 1);
  const current = inRun(passingChecks(), 2);
  assert.equal(policy.checksPass([...old, ...current], true), true);
  assert.equal(policy.checksPass([...inRun(passingChecks(), 1), ...current.slice(0, -1)], true), false);
  assert.equal(policy.checksPass([...current, ...inRun([
    {name: 'external CI', status: 'COMPLETED', conclusion: 'FAILURE'},
  ], 1, 'other workflow')], true), false);
  assert.equal(policy.checksPass([...current, {...current[0], databaseId: 299, conclusion: 'FAILURE'}], true), false);
  assert.equal(policy.checksPass([...current, {...current[0], databaseId: 199, conclusion: 'FAILURE'}], true), true);
});

test('merge uses squash, exact checked SHA, and explicitly starts post-merge CI', async () => {
  const args = mockMerge();
  assert.equal(await maybeMerge(args, 7), true);
  assert.equal(args.calls[0][1].merge_method, 'squash');
  assert.equal(args.calls[0][1].sha, 'abc');
  assert.equal(args.calls[1][1].workflow_id, 'ci.yml');
  assert.equal(args.calls[1][1].ref, 'main');
});

test('stale, blocked, conflicting, and truncated check results cannot merge', async () => {
  for (const overrides of [{headRefOid: 'different'}, {mergeStateStatus: 'BEHIND'},
    {mergeable: 'CONFLICTING'}, {reviewDecision: 'CHANGES_REQUESTED'}]) {
    const args = mockMerge(overrides);
    assert.equal(await maybeMerge(args, 7), false);
    assert.deepEqual(args.calls, []);
  }
  const args = mockMerge();
  args.state.commits.nodes[0].commit.statusCheckRollup.contexts.pageInfo.hasNextPage = true;
  assert.equal(await maybeMerge(args, 7), false);
});

test('hold added during check inspection prevents the merge', async () => {
  const args = mockMerge();
  const query = args.github.graphql;
  args.github.graphql = async () => {
    const result = await query();
    args.pr.labels.push({name: 'hold'});
    return result;
  };
  assert.equal(await maybeMerge(args, 7), false);
  assert.deepEqual(args.calls, []);
});

test('dependency eligibility must be verified for the current SHA', async () => {
  const args = mockMerge();
  args.pr.user.login = 'dependabot[bot]';
  args.pr.labels = [{name: 'automerge:dependencies'}];
  args.github.rest.repos = {getCombinedStatusForRef: async input => {
    assert.equal(input.ref, 'abc');
    return {data: {statuses: []}};
  }};
  assert.equal(await maybeMerge(args, 7), false);
  assert.deepEqual(args.calls, []);
  args.github.rest.repos.getCombinedStatusForRef = async () => ({data: {statuses: [{
    context: 'Dependabot policy', state: 'success', description: 'Verified patch/minor update',
    creator: {login: 'github-actions[bot]'},
  }]}});
  assert.equal(await maybeMerge(args, 7), true);
});

test('label reconciliation removes stale generated labels but keeps human controls', async () => {
  const calls = [];
  const pr = {...pull(), labels: [{name: 'release:major'}, {name: 'size:XL'}, {name: 'hold'}]};
  const github = {rest: {issues: {
    removeLabel: async args => calls.push(['remove', args.name]),
    addLabels: async args => calls.push(['add', args.labels]),
  }}};
  await reconcileLabels(github, {}, pr, ['nyssa_bench/datasets/export_json.py'], null);
  assert.ok(calls.some(([action, name]) => action === 'remove' && name === 'release:major'));
  assert.ok(!calls.some(([action, name]) => action === 'remove' && name === 'hold'));
  assert.ok(calls.find(([action]) => action === 'add')[1].includes('area:datasets'));
});

test('CI policy recognizes dispatched bot PRs and renamed container inputs', async () => {
  const outputs = {};
  const pr = {...pull(), changed_files: 1};
  let page = 0;
  const github = {paginate: async () => ++page === 1 ? [pr]
    : [{filename: 'archive/Dockerfile', previous_filename: 'docker/Dockerfile'}],
  graphql: async () => ({repository: {pullRequest: {headRefOid: 'abc'}}}),
  rest: {pulls: {list() {}, listFiles() {}, get: async () => ({data: pr})}}};
  await ciPolicy({github, context: {repo: {owner: 'owner', repo: 'repo'}, payload: {},
    ref: 'refs/heads/release-please--branches--main', sha: 'abc'},
  core: {info() {}, setOutput: (name, value) => { outputs[name] = value; }}});
  assert.equal(outputs.containers, 'true');
});

test('release automation refuses an unchecked current main', async () => {
  const args = {core: core(), context: {repo: {}, payload: {}}, github: {rest: {
    repos: {getBranch: async () => ({data: {commit: {sha: 'new'}}})},
    actions: {listWorkflowRuns: async () => ({data: {workflow_runs: []}})},
  }}};
  await assert.rejects(verifyMain(args), /must pass CI/);
  args.context.payload.workflow_run = {head_sha: 'old'};
  await assert.doesNotReject(verifyMain(args));
});

test('release bot dispatches CI and labels without relying on token-triggered events', async () => {
  const calls = [];
  const prior = {...process.env};
  try {
    process.env.RELEASE_PRS = '[{"number": 8}]';
    process.env.RELEASE_CREATED = 'true';
    process.env.RELEASE_TAG = 'v0.2.0';
    await dispatch({core: core(), context: {repo: {owner: 'owner', repo: 'repo'}}, github: {rest: {
      pulls: {get: async () => ({data: {...pull(), number: 8,
        head: {ref: 'release-please--branches--main', repo: {full_name: 'owner/repo'}}}})},
      actions: {createWorkflowDispatch: async args => calls.push(args)},
    }}});
    assert.deepEqual(calls.map(call => call.workflow_id), ['ci.yml', 'pr-automation.yml', 'release.yml']);
    assert.equal(calls[2].ref, 'v0.2.0');
  } finally {
    for (const key of ['RELEASE_PRS', 'RELEASE_CREATED', 'RELEASE_TAG']) {
      if (prior[key] === undefined) delete process.env[key]; else process.env[key] = prior[key];
    }
  }
});
