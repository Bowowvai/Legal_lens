# Supabase Auth (Google + Email) — Static HTML

This folder contains a minimal, static Tailwind UI wired to Supabase Auth:
- Google OAuth login
- Email/password sign-in and sign-up
- Forgot password email flow
- Reset password page (via magic link)

## Files
- `index.html` — Login page with Google and email flows.
- `forgot.html` — Form to request a reset email.
- `reset.html` — Sets a new password after clicking the email link.
- `config.example.js` — Template for your Supabase credentials. Copy to `config.js` and fill values.

## Setup
1. In the Supabase Dashboard, create a project. Copy your Project URL and anon key.
2. Copy `config.example.js` to `config.js` and set values:
   - `window.SUPABASE_URL = 'https://YOUR-REF.supabase.co'`
   - `window.SUPABASE_ANON_KEY = 'YOUR-ANON-KEY'`
   - Optionally change `POST_LOGIN_REDIRECT`.
3. Under Authentication → Providers → Google:
   - Enable Google, add your OAuth client credentials from Google Cloud Console.
   - Set Authorized redirect: `https://your-domain.tld/` (or `http://localhost:5500/` if serving locally).
4. Under Authentication → URL configuration:
   - Site URL: your domain or local dev URL (must match where you serve these files).
   - Redirect URLs: include `http://localhost:5500/` (or your domain), plus `http://localhost:5500/reset.html`.

## Local development
Use any static server (opening the file directly via `file://` will block OAuth redirects).

Example using Python:

```sh
# macOS
python3 -m http.server 5500
# then open http://localhost:5500/index.html
```

Or with Node:

```sh
npx serve -l 5500
```

## Notes
- Sign-up sends a confirmation email by default (can be configured in Supabase Auth settings).
- Forgot password sends a link that redirects to `reset.html` where the user sets a new password.
- The anon key is designed for client use. Do not expose the service role key on the client.
- To persist sessions across reloads, we additionally store the session in localStorage/sessionStorage based on the "Remember me" checkbox.
