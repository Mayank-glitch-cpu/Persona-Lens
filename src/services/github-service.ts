import { Octokit } from '@octokit/rest';
import { GitHubApiManager } from '../utils/github-api-manager';
import { GITHUB_TOKENS } from '../config';

export class GitHubService {
  private apiManager: GitHubApiManager;
  
  constructor() {
    this.apiManager = new GitHubApiManager(GITHUB_TOKENS);
  }

  async getRepositoryInfo(owner: string, repo: string) {
    return this.apiManager.executeRequest((octokit) => 
      octokit.repos.get({ owner, repo })
    );
  }

  async getRepositoryContributors(owner: string, repo: string) {
    return this.apiManager.executeRequest((octokit) => 
      octokit.repos.listContributors({ owner, repo })
    );
  }

  async getUserInfo(username: string) {
    return this.apiManager.executeRequest((octokit) => 
      octokit.users.getByUsername({ username })
    );
  }

  async getRepositoryCommits(owner: string, repo: string, options: { per_page?: number, page?: number } = {}) {
    return this.apiManager.executeRequest((octokit) => 
      octokit.repos.listCommits({ owner, repo, ...options })
    );
  }

  // Add other GitHub API methods as needed
}

// Create a singleton instance
export const githubService = new GitHubService();
