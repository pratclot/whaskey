import { defineConfig } from 'astro/config';

// Set GITHUB_REPO in .env or GitHub Actions to auto-configure base path and "Edit on GitHub" links
// e.g. GITHUB_REPO="username/blot"
const repo = process.env.GITHUB_REPO || 'pratclot/whaskey';
const repoName = repo.includes('/') ? repo.split('/')[1] : '';
const isGitHubActions = process.env.GITHUB_ACTIONS === 'true';

export default defineConfig({
  site: process.env.SITE_URL || (repo ? `https://${repo.split('/')[0]}.github.io` : 'http://localhost:4321'),
  base: isGitHubActions && repoName ? `/${repoName}` : '/',
});
