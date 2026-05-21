## Public Deployment

This folder is ready to deploy as a public Streamlit app.

### Files included

- `tbi_30d_streamlit_app.py`
- `requirements.txt`
- `tbi_30d_streamlit_artifacts/lasso_30d_risk_calculator.joblib`
- `tbi_30d_streamlit_artifacts/ui_metadata.json`

### Create a public link

1. Create a new GitHub repository.
2. Upload all files from this folder to that repository.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
4. Click `Create app`.
5. Select your GitHub repository.
6. Set the main file path to `tbi_30d_streamlit_app.py`.
7. Click `Deploy`.

After deployment, Streamlit will generate a public URL like:

`https://your-app-name.streamlit.app`

### Important

- Do not upload patient-level source data.
- This app package contains the trained model artifact, not the raw training dataset.
- If you change the app file name, update the Streamlit entrypoint during deployment.
