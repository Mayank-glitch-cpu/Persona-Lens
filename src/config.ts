// ...existing code...

// Support for multiple GitHub tokens
export const GITHUB_TOKENS = (process.env.GITHUB_TOKENS || process.env.GITHUB_TOKEN || '')
  .split(',')
  .map(token => token.trim())
  .filter(token => token.length > 0);

// Fallback to a single token if GITHUB_TOKENS isn't set
if (GITHUB_TOKENS.length === 0 && process.env.GITHUB_TOKEN) {
  GITHUB_TOKENS.push(process.env.GITHUB_TOKEN);
}

// ...existing code...
