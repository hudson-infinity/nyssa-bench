'use strict';

async function readMergeState(github, repo, number) {
  const result = await github.graphql(`query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) { pullRequest(number: $number) {
      headRefOid mergeStateStatus mergeable reviewDecision
      commits(last: 1) { nodes { commit { statusCheckRollup {
        contexts(first: 100) { pageInfo { hasNextPage } nodes {
          ... on CheckRun { name status conclusion checkSuite { app { slug } } }
          ... on StatusContext { context state }
        } }
      } } } }
    } }
  }`, {...repo, number});
  return result.repository.pullRequest;
}

module.exports = {readMergeState};
