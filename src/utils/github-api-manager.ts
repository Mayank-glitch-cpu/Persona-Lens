import { Octokit } from '@octokit/rest';
import { throttling } from '@octokit/plugin-throttling';

const ThrottledOctokit = Octokit.plugin(throttling);

interface TokenInfo {
  token: string;
  octokit: Octokit;
  remainingRequests: number;
  resetTime: number;
  isExhausted: boolean;
}

export class GitHubApiManager {
  private tokens: TokenInfo[] = [];
  private currentTokenIndex = 0;

  constructor(githubTokens: string[]) {
    if (!githubTokens || githubTokens.length === 0) {
      throw new Error('At least one GitHub token is required');
    }

    // Initialize tokens
    githubTokens.forEach(token => {
      const octokit = new ThrottledOctokit({
        auth: token,
        throttle: {
          onRateLimit: (retryAfter, options, octokit, retryCount) => {
            this.markTokenAsExhausted(token);
            return false; // Don't retry, we'll handle it with token rotation
          },
          onSecondaryRateLimit: (retryAfter, options, octokit) => {
            this.markTokenAsExhausted(token);
            return false; // Don't retry, we'll handle it with token rotation
          }
        }
      });

      this.tokens.push({
        token,
        octokit,
        remainingRequests: 5000, // Default limit
        resetTime: Date.now() + 3600000, // Default 1 hour from now
        isExhausted: false
      });
    });
  }

  private markTokenAsExhausted(token: string): void {
    const tokenInfo = this.tokens.find(t => t.token === token);
    if (tokenInfo) {
      tokenInfo.isExhausted = true;
      tokenInfo.remainingRequests = 0;
    }
  }

  private async updateRateLimits(tokenIndex: number): Promise<void> {
    try {
      const response = await this.tokens[tokenIndex].octokit.rateLimit.get();
      const { remaining, reset } = response.data.rate;
      
      this.tokens[tokenIndex].remainingRequests = remaining;
      this.tokens[tokenIndex].resetTime = reset * 1000; // Convert to milliseconds
      
      // Reset exhausted flag if we have requests available again
      if (remaining > 0 && Date.now() >= this.tokens[tokenIndex].resetTime) {
        this.tokens[tokenIndex].isExhausted = false;
      }
    } catch (error) {
      console.error(`Failed to check rate limits for token ${tokenIndex}:`, error);
    }
  }

  private getNextAvailableTokenIndex(): number {
    const now = Date.now();
    
    // Reset tokens that have passed their reset time
    this.tokens.forEach((token, index) => {
      if (token.isExhausted && now >= token.resetTime) {
        token.isExhausted = false;
        token.remainingRequests = 5000; // Reset to default
      }
    });
    
    // Find the next available token
    for (let i = 0; i < this.tokens.length; i++) {
      const index = (this.currentTokenIndex + i) % this.tokens.length;
      if (!this.tokens[index].isExhausted) {
        this.currentTokenIndex = index;
        return index;
      }
    }
    
    // If all tokens are exhausted, return the one that will reset first
    const nextToken = this.tokens.reduce((earliest, current, index) => {
      return current.resetTime < earliest.resetTime ? 
        { index, resetTime: current.resetTime } : 
        earliest;
    }, { index: 0, resetTime: Infinity });
    
    this.currentTokenIndex = nextToken.index;
    return nextToken.index;
  }

  public async executeRequest<T>(
    requestFn: (octokit: Octokit) => Promise<T>
  ): Promise<T> {
    // Try all tokens if needed
    for (let attempt = 0; attempt < this.tokens.length; attempt++) {
      const tokenIndex = this.getNextAvailableTokenIndex();
      
      try {
        // Periodically update rate limits
        if (attempt === 0 || this.tokens[tokenIndex].remainingRequests < 100) {
          await this.updateRateLimits(tokenIndex);
        }

        // Skip exhausted tokens
        if (this.tokens[tokenIndex].isExhausted) {
          continue;
        }

        // Execute the request
        const result = await requestFn(this.tokens[tokenIndex].octokit);
        
        // Decrement remaining requests estimate
        this.tokens[tokenIndex].remainingRequests--;
        
        return result;
      } catch (error: any) {
        if (error.status === 403 && 
            (error.message.includes('API rate limit exceeded') || 
             error.message.includes('secondary rate limit'))) {
          // Mark this token as exhausted
          this.tokens[tokenIndex].isExhausted = true;
          this.tokens[tokenIndex].remainingRequests = 0;
          
          // Try next token
          continue;
        }
        
        // For other errors, just throw
        throw error;
      }
    }
    
    // If we've tried all tokens and all are exhausted
    const earliestReset = Math.min(...this.tokens.map(t => t.resetTime));
    const waitTime = earliestReset - Date.now();
    
    throw new Error(`All GitHub API tokens are rate limited. Earliest reset in ${Math.ceil(waitTime / 60000)} minutes.`);
  }
}
