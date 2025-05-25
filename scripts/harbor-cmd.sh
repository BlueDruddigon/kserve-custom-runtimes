#!/bin/bash

# load secrets from `.env`
source .env

# first get all repositories that are available
curl -sk -X GET \
  -u "${HARBOR_USERNAME}:${HARBOR_PASSWORD}" \
  -H "Accept: application/json" \
  "https://${HARBOR_URL}/api/v2.0/projects/${PROJECT}/repositories" |
  jq -r '.[].name'

# get all artifacts within current repositories
DIGESTS=$(
  curl -sk -X GET \
    -u "${HARBOR_USERNAME}:${HARBOR_PASSWORD}" \
    -H "Accept: application/json" \
    "https://${HARBOR_URL}/api/v2.0/projects/${PROJECT}/repositories/${IMAGE}/artifacts" |
    jq -r '.[].digest'
)
# then delete them
for digest in ${DIGESTS}; do
  if [[ -n ${digest} ]]; then
    curl -sk -X DELETE -u "${HARBOR_USERNAME}:${HARBOR_PASSWORD}" "https://${HARBOR_URL}/api/v2.0/projects/${PROJECT}/repositories/${IMAGE}/artifacts/${digest}"
    echo "deleted ${REPO}@${digest}"
  fi
done
