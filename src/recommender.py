import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_jobs(user_skills, job_data_path, top_n=5):
    try:
        # Load job data
        jobs = pd.read_csv(job_data_path)

        # Validate column presence
        if 'required_skills' not in jobs.columns:
            raise KeyError("Missing required 'required_skills' column in job dataset.")

        # Drop rows with missing skills
        jobs = jobs[jobs['required_skills'].apply(lambda x: isinstance(x, str))].copy()
        jobs['required_skills'] = jobs['required_skills'].fillna("")

        # Vectorization
        vec = TfidfVectorizer()
        job_vecs = vec.fit_transform(jobs['required_skills'])
        user_vec = vec.transform([user_skills])

        # Cosine similarity
        sims = cosine_similarity(user_vec, job_vecs).flatten()
        top = sims.argsort()[-top_n:][::-1]

        # Add match %
        top_jobs = jobs.iloc[top].copy()
        top_jobs['Match %'] = (sims[top] * 100).round(2)

        # Return useful columns
        if 'job_post' in top_jobs.columns:
            return top_jobs[['job_post', 'required_skills','salary_offered', 'Match %']]
        else:
            return top_jobs[['required_skills', 'Match %']]

    except Exception as e:
        raise RuntimeError(f"Error while recommending jobs: {str(e)}")

