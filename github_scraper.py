import pandas as pd
import requests
from bs4 import BeautifulSoup
from github import Github
from github.GithubException import RateLimitExceededException, BadCredentialsException
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import json
import random
import socket
from requests.exceptions import RequestException, Timeout, ConnectionError
import urllib3
from collections import Counter

# Load environment variables
load_dotenv()

# Get GitHub tokens - handle both comma-separated and line-by-line formats
raw_tokens = os.getenv('GITHUB_TOKENS', os.getenv('GITHUB_TOKEN', ''))
# First try to split by comma for backward compatibility
if ',' in raw_tokens:
    GITHUB_TOKENS = raw_tokens.split(',')
else:
    # Otherwise split by newline for the new format
    GITHUB_TOKENS = raw_tokens.strip().split('\n')

# Clean up tokens
GITHUB_TOKENS = [token.strip() for token in GITHUB_TOKENS if token.strip()]

# Add the single token if it's not already included
single_token = os.getenv('GITHUB_TOKEN')
if (single_token and single_token.strip() not in GITHUB_TOKENS):
    GITHUB_TOKENS.append(single_token.strip())

if not GITHUB_TOKENS:
    raise ValueError("No GitHub tokens found. Please set GITHUB_TOKEN or GITHUB_TOKENS in your .env file")

print(f"Loaded {len(GITHUB_TOKENS)} GitHub API tokens")

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Check if there is an internet connection by trying to connect to Google's DNS server
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def wait_for_internet(max_retries=5, initial_delay=5):
    """
    Wait for internet connection to be available
    """
    for attempt in range(max_retries):
        if check_internet_connection():
            print("Internet connection established.")
            return True
        
        wait_time = initial_delay * (2 ** attempt)
        print(f"No internet connection. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
        time.sleep(wait_time)
    
    print("Failed to establish internet connection after multiple attempts.")
    return False

# Token manager class
class GitHubTokenManager:
    def __init__(self, tokens, max_validation_retries=3):
        self.tokens = []
        self.token_info = {}
        self.current_index = 0
        
        # Check internet connection before proceeding
        if not check_internet_connection():
            print("Warning: No internet connection detected. Waiting for connection...")
            if not wait_for_internet():
                print("Continuing without internet validation. Tokens will be validated when connection is available.")
                # Store tokens without validation for now
                for token in tokens:
                    self.tokens.append(token)
                    self.token_info[token] = {
                        'client': Github(token, timeout=30),
                        'username': 'unknown',
                        'remaining': 5000,
                        'reset_time': datetime.now().timestamp(),
                        'exhausted': False,
                        'valid': True,  # Assume valid until proven otherwise
                        'validated': False
                    }
                return
                
        # Validate tokens before using them
        print("Validating tokens...")
        for token in tokens:
            validated = False
            for attempt in range(max_validation_retries):
                try:
                    client = Github(token, timeout=30)
                    # Test the token with a simple API call
                    user = client.get_user()
                    print(f"✓ Token valid - authenticated as: {user.login}")
                    
                    self.tokens.append(token)
                    self.token_info[token] = {
                        'client': client,
                        'username': user.login,
                        'remaining': 5000,  # Default GitHub rate limit
                        'reset_time': datetime.now().timestamp(),
                        'exhausted': False,
                        'valid': True,
                        'validated': True
                    }
                    validated = True
                    break
                    
                except (urllib3.exceptions.MaxRetryError, ConnectionError, socket.gaierror) as e:
                    if "name resolution" in str(e).lower() or "temporary failure" in str(e).lower():
                        print(f"Network error while validating token: {str(e)}")
                        if attempt < max_validation_retries - 1:
                            wait_time = 5 * (2 ** attempt)
                            print(f"Possible DNS or network issue. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_validation_retries})")
                            time.sleep(wait_time)
                        else:
                            # If we've tried multiple times and still can't connect, store the token anyway
                            # and we'll validate it later when we try to use it
                            print(f"Adding token without validation due to network issues")
                            self.tokens.append(token)
                            self.token_info[token] = {
                                'client': client,
                                'username': 'unknown',
                                'remaining': 5000,
                                'reset_time': datetime.now().timestamp(),
                                'exhausted': False,
                                'valid': True,  # Assume valid until proven otherwise
                                'validated': False
                            }
                            validated = True
                    else:
                        print(f"✗ Error validating token: {str(e)}")
                
                except BadCredentialsException:
                    print(f"✗ Token invalid - authentication failed")
                    validated = True  # Mark as validated (though invalid)
                    break
                
                except Exception as e:
                    print(f"✗ Error validating token: {str(e)}")
                    if attempt < max_validation_retries - 1:
                        wait_time = 5 * (2 ** attempt)
                        print(f"Retrying validation in {wait_time} seconds... (Attempt {attempt+1}/{max_validation_retries})")
                        time.sleep(wait_time)
                    else:
                        validated = True  # Mark as validated (failed)
            
            # If we couldn't validate due to network issues but we're on the last attempt,
            # add the token anyway and mark it for validation when we use it
            if not validated:
                print(f"Adding token without validation due to persistent issues")
                self.tokens.append(token)
                self.token_info[token] = {
                    'client': Github(token, timeout=30),
                    'username': 'unknown',
                    'remaining': 5000,
                    'reset_time': datetime.now().timestamp(),
                    'exhausted': False,
                    'valid': True,  # Assume valid until proven otherwise
                    'validated': False
                }
        
        if not self.tokens:
            raise ValueError("No valid GitHub tokens found. Please check your tokens in the .env file or your network connection.")
        
        print(f"Added {len(self.tokens)} tokens (some may need validation when network is available)")
    
    def get_client(self):
        """Get the next available GitHub client"""
        if not self.tokens:
            raise ValueError("No valid tokens available")
        
        # Check if we have a network connection if there are unvalidated tokens
        if any(not self.token_info[token].get('validated', False) for token in self.tokens) and check_internet_connection():
            # Validate any unvalidated tokens
            for token in list(self.tokens):  # Use a copy of the list since we might modify it
                if not self.token_info[token].get('validated', False):
                    try:
                        client = self.token_info[token]['client']
                        user = client.get_user()
                        print(f"✓ Token now validated - authenticated as: {user.login}")
                        self.token_info[token]['username'] = user.login
                        self.token_info[token]['validated'] = True
                    except BadCredentialsException:
                        print(f"✗ Token invalid - authentication failed")
                        self.invalidate_token(token)
                    except Exception as e:
                        print(f"Failed to validate token: {str(e)}")
                        # Keep the token for now, we'll try again later
            
        # Check if any tokens have reset
        now = datetime.now().timestamp()
        for token, info in self.token_info.items():
            if info['exhausted'] and now > info['reset_time']:
                print(f"Token reset time reached. Resetting exhausted status.")
                info['exhausted'] = False
                info['remaining'] = 5000  # Reset to default
        
        # Try each token
        checked_tokens = 0
        while checked_tokens < len(self.tokens):
            try:
                token = self.tokens[self.current_index]
                info = self.token_info[token]
                
                # Move to next token for next call
                self.current_index = (self.current_index + 1) % len(self.tokens)
                checked_tokens += 1
                
                if not info['exhausted'] and info['valid']:
                    # Try to validate if not yet validated
                    if not info.get('validated', False) and check_internet_connection():
                        try:
                            client = info['client']
                            user = client.get_user()
                            print(f"✓ Token now validated - authenticated as: {user.login}")
                            info['username'] = user.login
                            info['validated'] = True
                        except BadCredentialsException:
                            print(f"✗ Token invalid - authentication failed")
                            self.invalidate_token(token)
                            continue
                        except Exception as e:
                            print(f"Failed to validate token: {str(e)}")
                            # Continue with the token anyway
                    
                    return info['client'], token
            except IndexError:
                # Handle the case where tokens list got modified during iteration
                if self.tokens:
                    self.current_index = 0
                    continue
                else:
                    raise ValueError("No valid tokens available")
        
        # If all tokens are exhausted, find the one that resets soonest
        min_reset_time = float('inf')
        token_to_use = None
        
        for token, info in self.token_info.items():
            if info['valid'] and info['reset_time'] < min_reset_time:
                min_reset_time = info['reset_time']
                token_to_use = token
        
        if not token_to_use:
            raise ValueError("No valid tokens available")
            
        wait_time = max(0, min_reset_time - now)
        if wait_time > 0:
            print(f"All tokens are exhausted. Waiting {wait_time/60:.1f} minutes for reset...")
            time.sleep(wait_time + 10)  # Add a small buffer
        
        # Reset token status after waiting
        self.token_info[token_to_use]['exhausted'] = False
        self.token_info[token_to_use]['remaining'] = 5000
        return self.token_info[token_to_use]['client'], token_to_use
    
    def update_rate_limit(self, token, exhausted=False):
        """Update rate limit info for a token"""
        try:
            client = self.token_info[token]['client']
            rate_limit = client.get_rate_limit()
            
            self.token_info[token]['remaining'] = rate_limit.core.remaining
            self.token_info[token]['reset_time'] = rate_limit.core.reset.timestamp()
            self.token_info[token]['exhausted'] = exhausted or rate_limit.core.remaining < 10
            
            print(f"Token has {rate_limit.core.remaining} requests remaining. Reset at {rate_limit.core.reset}")
        except BadCredentialsException:
            print(f"Token is no longer valid - marking as invalid")
            self.invalidate_token(token)
        except Exception as e:
            print(f"Error checking rate limit: {str(e)}")
            if "rate limit" in str(e).lower():
                self.token_info[token]['exhausted'] = True
                self.token_info[token]['remaining'] = 0
    
    def invalidate_token(self, token):
        """Mark a token as invalid (e.g., due to authentication failure)"""
        if token in self.token_info:
            print(f"Marking token as invalid")
            self.token_info[token]['valid'] = False
            if token in self.tokens:
                self.tokens.remove(token)
            
            if not self.tokens:
                raise ValueError("No valid GitHub tokens remaining. Please check your tokens in the .env file.")

# Try to create token manager, with retry for network issues
max_setup_retries = 3
for setup_attempt in range(max_setup_retries):
    try:
        token_manager = GitHubTokenManager(GITHUB_TOKENS)
        break
    except (ConnectionError, socket.gaierror, urllib3.exceptions.MaxRetryError) as e:
        if "name resolution" in str(e).lower() or "temporary failure" in str(e).lower():
            if setup_attempt < max_setup_retries - 1:
                wait_time = 10 * (2 ** setup_attempt)
                print(f"Network error during setup: {str(e)}")
                print(f"Retrying in {wait_time} seconds... (Attempt {setup_attempt+1}/{max_setup_retries})")
                time.sleep(wait_time)
            else:
                print(f"Failed to initialize after {max_setup_retries} attempts due to network issues.")
                raise ValueError("Failed to initialize token manager due to persistent network issues. Please check your internet connection.")
        else:
            raise

def get_user_data(username, max_retries=3, max_repos=1000, repo_timeout=60):
    """
    Get GitHub user data with improved handling for users with many repositories
    
    Args:
        username: GitHub username to process
        max_retries: Maximum number of retry attempts
        max_repos: Maximum number of repositories to process per user
        repo_timeout: Timeout in seconds for repository fetching
    """
    for attempt in range(max_retries):
        try:
            # Check internet connection before proceeding
            if not check_internet_connection():
                print("No internet connection. Waiting for connection...")
                if not wait_for_internet():
                    print("Failed to establish internet connection. Skipping this user.")
                    return None
            
            # Get the next available GitHub client
            g, current_token = token_manager.get_client()
            
            try:
                # Get user data through GitHub API
                user = g.get_user(username)
                
                # Get user repositories with rate limit handling and timeout protection
                try:
                    print(f"Fetching repositories for {username} (limiting to {max_repos} repos)...")
                    
                    # Use a timeout for getting repositories
                    import signal
                    
                    class TimeoutException(Exception):
                        pass
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutException("Repository fetch timed out")
                    
                    # Set timeout for repository fetching
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(repo_timeout)
                    
                    try:
                        # Limit the number of repositories to process
                        repos = []
                        repo_count = 0
                        
                        for repo in user.get_repos():
                            repos.append(repo)
                            repo_count += 1
                            if repo_count >= max_repos:
                                print(f"Reached maximum repository limit ({max_repos}) for {username}")
                                break
                        
                        # Calculate stats based on fetched repos
                        total_stars = sum(repo.stargazers_count for repo in repos)
                        total_forks = sum(repo.forks_count for repo in repos)
                        
                        # Reset the alarm
                        signal.alarm(0)
                        
                    except TimeoutException:
                        print(f"Timeout fetching repositories for {username} after {repo_timeout} seconds")
                        # Use whatever repos we managed to get before timeout
                        total_stars = sum(repo.stargazers_count for repo in repos) if repos else 0
                        total_forks = sum(repo.forks_count for repo in repos) if repos else 0
                    
                    # Enhanced language statistics collection
                    language_stats = {}
                    for repo in repos:
                        if repo.language:
                            # Weight languages by repository size or stars
                            weight = max(1, repo.stargazers_count)  # Use stars as weight, minimum of 1
                            language_stats[repo.language] = language_stats.get(repo.language, 0) + weight
                    
                    # Sort languages by usage and get top 10
                    total_weight = sum(language_stats.values())
                    
                    # Only process if we have language data
                    if total_weight > 0:
                        # Sort languages by weight and take top 10
                        top_languages = sorted(language_stats.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        # Calculate percentages
                        top_languages_with_pct = []
                        for lang, weight in top_languages:
                            percentage = (weight / total_weight) * 100
                            top_languages_with_pct.append({
                                'language': lang,
                                'percentage': round(percentage, 2)
                            })
                        
                        # Store as JSON string
                        language_data = json.dumps(top_languages_with_pct)
                        most_used_language = top_languages[0][0] if top_languages else "None"
                        
                        # Create individual language fields for top 10 languages
                        language_fields = {}
                        for i, lang_data in enumerate(top_languages_with_pct, 1):
                            if i <= 10:  # Ensure we only keep top 10
                                lang_name = lang_data['language']
                                lang_pct = lang_data['percentage']
                                language_fields[f'language_{i}'] = lang_name
                                language_fields[f'language_{i}_pct'] = lang_pct
                                
                        # Add empty values for missing languages (if less than 10)
                        for i in range(len(top_languages_with_pct) + 1, 11):
                            language_fields[f'language_{i}'] = ""
                            language_fields[f'language_{i}_pct'] = 0.0
                    else:
                        language_data = "[]"
                        most_used_language = "None"
                        # Empty language fields
                        language_fields = {}
                        for i in range(1, 11):
                            language_fields[f'language_{i}'] = ""
                            language_fields[f'language_{i}_pct'] = 0.0
                        
                except RateLimitExceededException:
                    print(f"Rate limit exceeded while processing repos for {username}")
                    token_manager.update_rate_limit(current_token, exhausted=True)
                    
                    if attempt < max_retries - 1:
                        print(f"Retrying with another token... (Attempt {attempt+2}/{max_retries})")
                        continue
                    else:
                        total_stars = -1
                        total_forks = -1
                        most_used_language = "Rate Limited"
                        language_data = "[]"
                        # Empty language fields
                        language_fields = {}
                        for i in range(1, 11):
                            language_fields[f'language_{i}'] = ""
                            language_fields[f'language_{i}_pct'] = 0.0
                finally:
                    # Ensure alarm is reset even if there's an exception
                    signal.alarm(0)
            except BadCredentialsException:
                print(f"Bad credentials error with token for user {username}")
                token_manager.invalidate_token(current_token)
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
            except (urllib3.exceptions.MaxRetryError, ConnectionError, socket.gaierror) as e:
                if "name resolution" in str(e).lower() or "temporary failure" in str(e).lower():
                    print(f"Network error while processing user {username}: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = 10 * (2 ** attempt)
                        print(f"Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Failed after {max_retries} attempts due to network issues.")
                        return None
                else:
                    raise
            
            # Scrape additional data using BeautifulSoup with delay
            time.sleep(1)  # Politeness delay
            
            # Add retry logic and increased timeout for HTTP requests
            session = requests.Session()
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
            response = session.get(f'https://github.com/{username}', timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get contribution data
            contributions = soup.find('h2', class_='f4 text-normal mb-2')
            contributions = contributions.text.strip() if contributions else "0"
            
            # Collect profile links
            profile_links = []
            for link in soup.find_all('a', class_='Link--primary'):
                if 'http' in link.get('href', ''):
                    profile_links.append(link['href'])
            
            # Create the basic user data dictionary
            user_data = {
                'username': username,
                'name': user.name or '',
                'bio': user.bio or '',
                'profile_links': ','.join(profile_links),
                'public_repos': user.public_repos,
                'total_stars': total_stars,
                'total_forks': total_forks,
                'most_used_language': most_used_language,
                'language_stats': language_data,  # Keep original JSON for backwards compatibility
                'followers': user.followers,
                'following': user.following,
                'contributions_last_year': contributions,
                'account_created': user.created_at.strftime('%Y-%m-%d'),
                'location': user.location or '',
                'company': user.company or '',
                'email': user.email or '',
                'blog': user.blog or '',
                'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add language fields to user data
            user_data.update(language_fields)
            
            return user_data
            
        except (RequestException, Timeout, ConnectionError, socket.gaierror, urllib3.exceptions.MaxRetryError) as e:
            if "name resolution" in str(e).lower() or "temporary failure" in str(e).lower():
                print(f"Network error while processing user {username}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 10 * (2 ** attempt)
                    print(f"Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts due to network issues.")
                    return None
            else:
                if attempt < max_retries - 1:
                    # Add exponential backoff for retries
                    wait_time = (2 ** attempt) + random.random()
                    print(f"Network error for user {username}: {str(e)}. Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Network error: Failed to process user {username} after {max_retries} attempts: {str(e)}")
                    return None
        except BadCredentialsException:
            print(f"Bad credentials error with token for user {username}")
            token_manager.invalidate_token(current_token)
            if attempt < max_retries - 1:
                continue
            else:
                return None
        except Exception as e:
            print(f"Error processing user {username}: {str(e)}")
            if "bad credentials" in str(e).lower() or "401" in str(e):
                token_manager.invalidate_token(current_token)
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
            elif "rate limit" in str(e).lower() and attempt < max_retries - 1:
                # Mark token as exhausted and try another
                token_manager.update_rate_limit(current_token, exhausted=True)
                print(f"Switching to another token due to rate limit")
                continue
            return None

def save_progress(processed_users):
    with open('github-users/progress.json', 'w') as f:
        json.dump(processed_users, f)

def load_progress():
    try:
        with open('github-users/progress.json', 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def find_next_user_index(github_logins, processed_users):
    """Find the index of the next user to process after a crash"""
    for idx, login in enumerate(github_logins):
        if login in processed_users:
            continue
        return idx  # Return the first unprocessed user
    return 0  # Default to beginning if all processed

def main():
    try:
        # Try to load data with retries for network issues
        for attempt in range(3):
            try:
                # Read GitHub usernames from CSV
                df = pd.read_csv('github-users/github_users.csv')
                github_logins = df['login'].tolist()  # Changed from 'github_url' to 'login'
                break
            except Exception as e:
                if attempt < 2:
                    print(f"Error loading CSV: {str(e)}. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise
        
        # Load progress
        processed_users = load_progress()
        print(f"Resuming from {len(processed_users)} previously processed users")
        
        # Collect data for each user
        all_data = []
        
        # Load existing data if available
        output_file = f'github-users/github_profiles_{datetime.now().strftime("%Y%m%d")}.csv'
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            all_data = existing_df.to_dict('records')
        
        # Find next user to process (starting point)
        start_idx = find_next_user_index(github_logins, processed_users)
        
        for idx, login in enumerate(github_logins[start_idx:], start=start_idx):
            if login in processed_users:
                print(f"Skipping already processed user: {login}")
                continue
                
            print(f"Processing user: {login}")
            user_data = get_user_data(login)
            
            if user_data:
                all_data.append(user_data)
                processed_users.add(login)
                
                # Save progress after each successful scrape
                save_progress(list(processed_users))
                
                # Update CSV file
                result_df = pd.DataFrame(all_data)
                result_df.to_csv(output_file, index=False)
            else:
                print(f"Skipping user {login} due to errors, will try again in next run")
                
            # Variable delay between requests
            delay = 2 + random.random() * 2  # 2-4 seconds
            print(f"Waiting {delay:.1f} seconds before next request...")
            time.sleep(delay)
        
        print(f"Completed processing {len(processed_users)} users")
        print(f"Data saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Save progress even if we encounter an error
        if 'processed_users' in locals():
            save_progress(list(processed_users))

if __name__ == "__main__":
    main()