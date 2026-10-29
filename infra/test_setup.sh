#!/usr/bin/env bash
# Dry-runs infra/setup.sh against stub az/gh (no Azure calls) to catch shell
# errors and the az argument mistakes we've hit. Run: bash infra/test_setup.sh
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
stubs=$(mktemp -d)
log="$stubs/calls.log"
trap 'rm -rf "$stubs"' EXIT

cat > "$stubs/az" <<'EOF'
#!/usr/bin/env bash
echo "az $*" >> "$CALLS"
case "$*" in
  # az webapp config set has no --https-only; the real CLI rejects it.
  "webapp config set"*--https-only*) echo "unrecognized arguments: --https-only" >&2; exit 2 ;;
  # By-name/appId assignee races AAD replication right after sp create.
  "role assignment create"*"--assignee "*) echo "principal does not exist" >&2; exit 1 ;;
  "account show"*) echo "00000000-0000-0000-0000-000000000000" ;;
  "ad app create"*) echo "11111111-1111-1111-1111-111111111111" ;;
  "ad sp create"*) echo "22222222-2222-2222-2222-222222222222" ;;
esac
EOF
printf '#!/usr/bin/env bash\necho "gh $*" >> "$CALLS"\n' > "$stubs/gh"
chmod +x "$stubs/az" "$stubs/gh"

out=$(CALLS="$log" PATH="$stubs:$PATH" bash "$here/setup.sh" 2>&1)
rc=$?

fail() { echo "FAIL: $1"; echo "$out" | tail -5; exit 1; }
[ $rc -eq 0 ] || fail "setup.sh exited $rc"
grep -q "^az webapp create .*--https-only true" "$log" || fail "web apps not created https-only"
grep -q "^az role assignment create .*--assignee-object-id 22222222-2222-2222-2222-222222222222 --assignee-principal-type ServicePrincipal" "$log" \
  || fail "role assignment not by service principal object id"
[ "$(grep -c '^gh variable set' "$log")" -eq 5 ] || fail "expected 5 gh variable set calls"
# App Service evals the startup command inside single quotes: quotes/parens break it (exit 127).
startup=$(grep -o -- "--startup-file .*" "$log" | sed 's/^--startup-file //; s/ --generic-configurations.*//')
[ -n "$startup" ] || fail "no startup command set"
case "$startup" in *[\'\"\(\)]*) fail "startup command has quotes/parens: $startup" ;; esac
echo "$out" | grep -q "Done." || fail "no completion message"
echo "setup.sh dry run ok ($(wc -l < "$log") calls)"
