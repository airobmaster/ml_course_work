# Contributing

This started as a course-work project, but suggestions and improvements are welcome.

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies: `pip install -r requirements.txt`.
3. Create a feature branch: `git checkout -b feature/your-change`.

## Making changes

- Keep pull requests focused on a single change (a bug fix, a feature, a doc update).
- Follow the existing code style in the file you're editing.
- If you change the model, preprocessing, or feature engineering, regenerate the dashboard
  artifacts before testing (`scripts/train_model.py`, `scripts/generate_feature_metadata.py`,
  `dashboard/scripts/build_db.py` — see [README.md](README.md#how-to-run-the-app)).
- If you add or modify a dashboard page, run the app locally and click through it before
  submitting (`streamlit run dashboard/app.py`).

## Submitting

1. Commit your changes with a clear, descriptive message.
2. Push to your fork and open a pull request against `main`.
3. Describe what changed and why in the pull request description.

## Reporting issues

Open an issue describing the problem, the steps to reproduce it, and what you expected to
happen instead.
