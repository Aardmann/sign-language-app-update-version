(function () {
  const client = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

  const PAYSTACK_PUBLIC_KEY = 'pk_live_be7ac128a98aa85c216b52f64f1ad5523bd3193e';

  const VERIFY_TOPUP_URL = 'https://urgrewshuexdllzaqlud.supabase.co/functions/v1/verify-topup';

  const TRIAL_MS  = 20 * 60 * 1000;
  const TRIAL_KEY = 'sld_trial_start';

  // Cost per letter detected (GHS)
  const COST_PER_LETTER = 0.02;

  // Quick-select top-up amounts shown in the modal (GHS)
  const TOPUP_PRESETS = [5, 10, 20, 50, 100];

  const GREETINGS = [
    'Welcome back', 'Good to see you', 'Hey there', 'Howdy',
    'Glad you\'re here', 'Hello again', 'Welcome', 'Nice to see you',
    'Hi there', 'Great to have you back', 'Hullayy!! it\'s'
  ];

  const SIGNINGREATINGS = [
    'Welcome back', 'Good to see you again', 'Hey there', 'Howdy',
    'Glad you\'re back', 'Hello again', 'Akwaaba', 'Obaake',
    'Hi there', 'Great to have you back', 'Hullayyy!!'
  ];

  // ── Trial timer ──────────────────────────────────────────────────
  function getTrialStart() {
    const raw = localStorage.getItem(TRIAL_KEY);
    return raw ? parseInt(raw, 10) : null;
  }

  function startTrialIfNeeded() {
    let start = getTrialStart();
    if (!start) {
      start = Date.now();
      localStorage.setItem(TRIAL_KEY, String(start));
    }
    return start;
  }

  function getTrialRemainingMs() {
    const start = getTrialStart();
    if (!start) return TRIAL_MS;
    const elapsed = Date.now() - start;
    return Math.max(0, TRIAL_MS - elapsed);
  }

  function isTrialExpired() {
    const start = getTrialStart();
    if (!start) return false;
    return getTrialRemainingMs() <= 0;
  }

  function clearTrial() { localStorage.removeItem(TRIAL_KEY); }

  function formatRemaining(ms) {
    const totalSec = Math.ceil(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  // ── Session / profile ────────────────────────────────────────────
  async function getSession() {
    const { data, error } = await client.auth.getSession();
    if (error) return null;
    return data.session;
  }

  async function getProfile(userId) {
    const { data, error } = await client
      .from('profiles')
      .select('username, full_name')
      .eq('id', userId)
      .single();
    if (error) return null;
    return data;
  }

  async function signOut() {
    await client.auth.signOut();
    window.location.href = 'index.html';
  }

  function singnInGreeting() {
    return SIGNINGREATINGS[Math.floor(Math.random() * SIGNINGREATINGS.length)];
  }

  function randomGreeting() {
    return GREETINGS[Math.floor(Math.random() * GREETINGS.length)];
  }

  function initials(name) {
    if (!name) return '?';
    const parts = name.trim().split(/\s+/);
    const letters = parts.length > 1
      ? parts[0][0] + parts[parts.length - 1][0]
      : parts[0].slice(0, 2);
    return letters.toUpperCase();
  }

  async function redirectIfLoggedIn() {
    const session = await getSession();
    if (session) window.location.href = 'app.html';
  }

  // ── Credits ──────────────────────────────────────────────────────

  async function fetchBalance() {
    const { data, error } = await client
      .from('credits')
      .select('balance')
      .single();
    if (error || !data) return null;
    return parseFloat(data.balance);
  }

  // Deduct via tamper-proof server-side RPC — client cannot UPDATE credits directly
  async function deductCredits(amount) {
    const { data, error } = await client.rpc('deduct_credits', { p_amount: amount });
    if (error) return { ok: false, reason: error.message };
    return data;
  }

  function makePaystackRef() {
    return 'SLD-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8).toUpperCase();
  }

  // ── Paystack popup ───────────────────────────────────────────────
  async function openTopupPopup(amountGHS, userEmail, onSuccess) {
    // Guard: make sure the Paystack script actually loaded
    if (typeof PaystackPop === 'undefined') {
      alert('Paystack could not be loaded. Please check your internet connection and try again.');
      return;
    }

    const session = await getSession();
    if (!session) { window.location.href = 'signin.html'; return; }

    const amountPesewas = Math.round(amountGHS * 100); // GHS pesewas

    let handler;
    try {
      handler = PaystackPop.setup({
        key:      PAYSTACK_PUBLIC_KEY,
        email:    userEmail,
        amount:   amountPesewas,
        currency: 'GHS',
        ref:      makePaystackRef(),
        label:    'Sign Language Detector Credits',
        metadata: {
          custom_fields: [
            { display_name: 'Product',      variable_name: 'product',    value: 'SLD Credits' },
            { display_name: 'Amount (GHS)', variable_name: 'amount_ghs', value: String(amountGHS) }
          ]
        },

        // Paystack requires a plain synchronous function — no async allowed.
        // All async work goes in .then() chains.
        callback: function (response) {
          if (response.status !== 'success') return;

          const ref = response.reference;

          // 1. Call our Edge Function to independently verify the payment with
          //    Paystack's API and apply credits + flip status to 'success'.
          //    This is the authoritative credit application — no webhook delay.
          fetch(VERIFY_TOPUP_URL, {
            method:  'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer ' + session.access_token,
            },
            body: JSON.stringify({ reference: ref }),
          })
          .then(function(res) { return res.json(); })
          .then(function(result) {
            if (result.ok) {
              // Credits applied — fire the UI success handler with the new balance
              onSuccess(amountGHS, ref, result.balance);
            } else {
              console.error('verify-topup failed:', result.reason);
              // Still fire onSuccess so the UI isn't stuck; balance will
              // be re-fetched from the DB inside the handler
              onSuccess(amountGHS, ref, null);
            }
          })
          .catch(function(err) {
            console.error('verify-topup fetch error:', err);
            onSuccess(amountGHS, ref, null);
          });

          // NOTE: credit_topups is written exclusively by the verify-topup
          // Edge Function (via the apply_topup RPC), inside the same
          // transaction as the credits update. Do NOT insert into
          // credit_topups from the client — doing so races with the
          // Edge Function and can mark a row 'success' before credits
          // are actually applied, which makes apply_topup's idempotency
          // check wrongly skip the real update. It's also a security
          // hole, since it lets an authenticated client claim a 'success'
          // row without ever having the payment verified server-side.
        },

        onClose: function () {
          // User dismissed without paying — nothing to do.
        }
      });
    } catch (err) {
      console.error('PaystackPop.setup error:', err);
      alert('Could not initialise the payment popup. Please try again.');
      return;
    }

    handler.openIframe();
  }

  // ── Top-up modal ─────────────────────────────────────────────────
  function ensureTopupModal() {
    if (document.getElementById('topupModal')) return;

    const presetButtons = TOPUP_PRESETS.map(amt =>
      `<button class="topup-preset" data-amt="${amt}">GHS ${amt}</button>`
    ).join('');

    document.body.insertAdjacentHTML('beforeend', `
      <div class="modal-overlay" id="topupModal">
        <div class="modal-box topup-modal-box">
          <button class="modal-close" id="topupModalClose"><i class="fa-solid fa-xmark"></i></button>
          <div class="modal-icon" style="background:rgba(34,197,94,0.12);color:#22c55e">
            <i class="fa-solid fa-coins"></i>
          </div>
          <h3>Top Up Credits</h3>
          <p style="font-size:13px;color:var(--muted);margin-bottom:18px">
            Choose an amount or enter your own.
          </p>

          <div class="topup-presets" id="topupPresets">${presetButtons}</div>

          <div class="topup-custom">
            <label style="font-size:12px;color:var(--muted);margin-bottom:6px;display:block">
              Or enter custom amount (GHS)
            </label>
            <div style="display:flex;gap:8px">
              <input
                type="number"
                id="topupCustomAmt"
                min="1"
                placeholder="e.g. 25"
                style="
                  flex:1;padding:10px 12px;
                  background:var(--surface2);border:1px solid var(--border);
                  border-radius:9px;color:var(--text);font-size:14px;
                  font-family:inherit;outline:none;
                "
              />
            </div>
          </div>

          <div id="topupSelectedDisplay" style="
            text-align:center;font-size:13px;color:var(--muted);
            margin:14px 0 4px;min-height:20px;
          "></div>

          <button class="btn btn-primary btn-block" id="topupPayBtn" style="margin-top:10px">
            <i class="fa-brands fa-cc-visa"></i> Pay with Paystack
          </button>
          <button class="modal-dismiss" id="topupCancelBtn">Cancel</button>
        </div>
      </div>

      <style>
        .topup-modal-box { max-width: 380px; }
        .topup-presets {
          display: flex; flex-wrap: wrap; gap: 8px;
          justify-content: center; margin-bottom: 16px;
        }
        .topup-preset {
          padding: 8px 16px; border-radius: 9px;
          background: var(--surface2); border: 1px solid var(--border);
          color: var(--text); font-size: 13px; font-weight: 600;
          cursor: pointer; transition: all 0.15s; font-family: inherit;
        }
        .topup-preset:hover    { border-color: #22c55e; color: #22c55e; }
        .topup-preset.selected { background: #22c55e; color: #fff; border-color: #22c55e; }

        /* Credit badge in nav */
        .credit-badge {
          display: flex; align-items: center; gap: 6px;
          background: var(--surface2); border: 1px solid var(--border);
          border-radius: 10px; padding: 6px 12px;
          font-size: 13px; font-weight: 600; color: var(--text);
          cursor: default;
        }
        .credit-badge i { color: #22c55e; font-size: 12px; }
        .credit-badge.low   { border-color: rgba(240,160,51,0.5); color: var(--orange); }
        .credit-badge.low i { color: var(--orange); }
        .credit-badge.empty { border-color: rgba(226,75,74,0.5);  color: var(--red); }
        .credit-badge.empty i { color: var(--red); }

        /* Top-up button always in nav */
        .btn-topup {
          padding: 7px 14px; border-radius: 9px;
          background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3);
          color: #22c55e; font-size: 13px; font-weight: 600;
          cursor: pointer; transition: all 0.2s; font-family: inherit;
          display: flex; align-items: center; gap: 6px;
        }
        .btn-topup i { font-size: 12px; }
        .btn-topup:hover { background: rgba(34,197,94,0.22); }
      </style>
    `);

    // ── Preset button wiring ──────────────────────────────────────
    let selectedAmt = null;

    document.querySelectorAll('.topup-preset').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.topup-preset').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        selectedAmt = parseFloat(btn.dataset.amt);
        document.getElementById('topupCustomAmt').value = '';
        document.getElementById('topupSelectedDisplay').textContent =
          `You'll be charged GHS ${selectedAmt.toFixed(2)}`;
      });
    });

    document.getElementById('topupCustomAmt').addEventListener('input', e => {
      document.querySelectorAll('.topup-preset').forEach(b => b.classList.remove('selected'));
      selectedAmt = parseFloat(e.target.value) || null;
      document.getElementById('topupSelectedDisplay').textContent =
        selectedAmt ? `You'll be charged GHS ${selectedAmt.toFixed(2)}` : '';
    });

    document.getElementById('topupModalClose').addEventListener('click', dismissTopupModal);
    document.getElementById('topupCancelBtn').addEventListener('click', dismissTopupModal);

    // ── Pay button ────────────────────────────────────────────────
    document.getElementById('topupPayBtn').addEventListener('click', async () => {
      const amt = selectedAmt || parseFloat(document.getElementById('topupCustomAmt').value);
      if (!amt || amt < 1) {
        alert('Please select or enter an amount of at least GHS 1.');
        return;
      }

      const session = await getSession();
      if (!session) { window.location.href = 'signin.html'; return; }

      // IMPORTANT: do NOT dismiss the modal before calling openIframe().
      // Dismissing first makes the browser treat the iframe as an unprompted
      // popup (no longer in a trusted click handler chain) and blocks it.
      // The modal stays open; Paystack renders on top of it.
      openTopupPopup(amt, session.user.email, async (paidAmt, ref, newBalance) => {
        // Payment confirmed + credits applied by Edge Function
        dismissTopupModal();

        // If the Edge Function returned the new balance directly, use it
        // immediately. Otherwise fetch it from the DB.
        let bal = newBalance;
        if (bal === null || bal === undefined) {
          bal = await fetchBalance();
        }

        if (bal !== null) {
          updateCreditBadge(bal);
          window.dispatchEvent(new CustomEvent('credits:updated', { detail: { balance: bal } }));
        }

        alert(`GHS ${paidAmt.toFixed(2)} added to your credits!`);
      });
    });
  }

  function showTopupModal() {
    ensureTopupModal();
    // Reset selection state each time the modal opens
    document.querySelectorAll('.topup-preset').forEach(b => b.classList.remove('selected'));
    const customInput = document.getElementById('topupCustomAmt');
    if (customInput) customInput.value = '';
    const display = document.getElementById('topupSelectedDisplay');
    if (display) display.textContent = '';
    document.getElementById('topupModal').classList.add('show');
  }

  function dismissTopupModal() {
    const modal = document.getElementById('topupModal');
    if (modal) modal.classList.remove('show');
  }

  // ── Credit badge rendering ────────────────────────────────────────
  function updateCreditBadge(balance) {
    const badge = document.getElementById('creditBadge');
    if (!badge) return;

    const icon = document.createElement('i');
    icon.className = 'fa-solid fa-coins';

    badge.replaceChildren(icon, document.createTextNode(` GHS ${balance.toFixed(2)}`));

    badge.className = 'credit-badge';
    if (balance <= 0)   badge.classList.add('empty');
    else if (balance < 2) badge.classList.add('low');
  }

  window.Auth = {
    client,
    TRIAL_MS,
    COST_PER_LETTER,
    getTrialStart,
    startTrialIfNeeded,
    getTrialRemainingMs,
    isTrialExpired,
    clearTrial,
    formatRemaining,
    getSession,
    getProfile,
    signOut,
    randomGreeting,
    initials,
    redirectIfLoggedIn,
    singnInGreeting,
    // Credits API
    fetchBalance,
    deductCredits,
    showTopupModal,
    dismissTopupModal,
    updateCreditBadge,
  };
})();