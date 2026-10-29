#!/usr/bin/env bash
# Creates every Azure resource PokeGuess needs, in one resource group, and wires
# GitHub Actions to deploy to it (OIDC, no stored Azure secrets).
#
#   az login && gh auth login      # once
#   bash infra/setup.sh            # optional: RG=... LOCATION=... SUFFIX=... bash infra/setup.sh
#
# Cost: one B1 Linux App Service plan (~$13/month, hosts both apps) + Azure SQL
# free offer (serverless, auto-pauses when the monthly free limit is used up).
# The Azure free trial's $200 credit expires after 30 days; after that B1 bills
# pay-as-you-go. Tear everything down with:
#
#   az group delete -n "$RG" --yes
#
# then re-run this script to recreate it (new SUFFIX → new hostnames).
set -euo pipefail

RG=${RG:-pokeguess-rg}
LOCATION=${LOCATION:-eastus}
# Re-running after a partial failure? Pass the same SUFFIX so nothing is duplicated.
SUFFIX=${SUFFIX:-$(openssl rand -hex 3 | cut -c1-5)}
REPO=${REPO:-Joabera/PokeGuess}

PLAN=pokeguess-plan
WEB=pokeguess-web-$SUFFIX
INFERENCE=pokeguess-inference-$SUFFIX
SQL_SERVER=pokeguess-sql-$SUFFIX
SQL_DB=pokeguess
SQL_ADMIN=pokeguessadmin
SQL_PASSWORD="$(openssl rand -base64 24 | tr -d '/+=')Aa1!"
API_KEY=$(openssl rand -hex 32)

echo "==> Resource group $RG ($LOCATION), suffix $SUFFIX"
az group create -n "$RG" -l "$LOCATION" -o none

echo "==> App Service plan (B1 Linux) + two web apps"
az appservice plan create -g "$RG" -n "$PLAN" --is-linux --sku B1 -o none
az webapp create -g "$RG" -p "$PLAN" -n "$WEB" --runtime "DOTNETCORE:10.0" --https-only true -o none
az webapp create -g "$RG" -p "$PLAN" -n "$INFERENCE" --runtime "PYTHON:3.12" --https-only true -o none
for app in "$WEB" "$INFERENCE"; do
  az webapp config set -g "$RG" -n "$app" --always-on true -o none
done

echo "==> Inference app settings"
az webapp config set -g "$RG" -n "$INFERENCE" -o none \
  --startup-file "gunicorn --bind=0.0.0.0 --timeout 120 --workers 1 'app:create_app()'" \
  --generic-configurations '{"healthCheckPath": "/health"}'
az webapp config appsettings set -g "$RG" -n "$INFERENCE" -o none --settings \
  INFERENCE_API_KEY="$API_KEY" LEGACY_TRUST=0.55 SCM_DO_BUILD_DURING_DEPLOYMENT=true

echo "==> Azure SQL (free offer)"
az sql server create -g "$RG" -n "$SQL_SERVER" -l "$LOCATION" \
  --admin-user "$SQL_ADMIN" --admin-password "$SQL_PASSWORD" -o none
az sql server firewall-rule create -g "$RG" -s "$SQL_SERVER" -n AllowAzureServices \
  --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0 -o none
az sql db create -g "$RG" -s "$SQL_SERVER" -n "$SQL_DB" -o none \
  --edition GeneralPurpose --compute-model Serverless --family Gen5 --capacity 2 \
  --use-free-limit --free-limit-exhaustion-behavior AutoPause

echo "==> Web app settings"
CONN="Server=tcp:$SQL_SERVER.database.windows.net,1433;Database=$SQL_DB;User ID=$SQL_ADMIN;Password=$SQL_PASSWORD;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;"
az webapp config connection-string set -g "$RG" -n "$WEB" -t SQLAzure -o none --settings Default="$CONN"
az webapp config appsettings set -g "$RG" -n "$WEB" -o none --settings \
  Database__Provider=SqlServer \
  Inference__BaseUrl="https://$INFERENCE.azurewebsites.net" \
  Inference__ApiKey="$API_KEY"

echo "==> GitHub Actions OIDC identity"
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
CLIENT_ID=$(az ad app create --display-name "pokeguess-github-$SUFFIX" --query appId -o tsv)
SP_OBJECT_ID=$(az ad sp create --id "$CLIENT_ID" --query id -o tsv)
az ad app federated-credential create --id "$CLIENT_ID" -o none --parameters "{
  \"name\": \"github-main\",
  \"issuer\": \"https://token.actions.githubusercontent.com\",
  \"subject\": \"repo:$REPO:ref:refs/heads/main\",
  \"audiences\": [\"api://AzureADTokenExchange\"]
}"
# Assign by object id: a just-created principal may not be resolvable by appId yet.
az role assignment create --assignee-object-id "$SP_OBJECT_ID" --assignee-principal-type ServicePrincipal \
  --role Contributor \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RG" -o none

echo "==> GitHub repo variables ($REPO)"
gh variable set AZURE_CLIENT_ID --repo "$REPO" --body "$CLIENT_ID"
gh variable set AZURE_TENANT_ID --repo "$REPO" --body "$TENANT_ID"
gh variable set AZURE_SUBSCRIPTION_ID --repo "$REPO" --body "$SUBSCRIPTION_ID"
gh variable set AZURE_WEBAPP_WEB --repo "$REPO" --body "$WEB"
gh variable set AZURE_WEBAPP_INFERENCE --repo "$REPO" --body "$INFERENCE"

echo
echo "Done. Push to main (or run the Deploy workflow) to ship."
echo "  Web:       https://$WEB.azurewebsites.net"
echo "  Inference: https://$INFERENCE.azurewebsites.net/health"
