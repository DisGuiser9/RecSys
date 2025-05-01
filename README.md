# Book Recommendation System

## Project Overview

This is a Flask-based book recommendation system utilizing the KNN collaborative filtering algorithm. Users can receive personalized book recommendations through this system. The frontend interface is inspired by the Taobao style, featuring a dark background, with read books displayed on the left, recommended books on the right, and user information and recommendation reasons at the top.

## Installation Guide

### Prerequisites

- Python 3.x
- Flask
- Pandas
- Numpy
- Scikit-learn

### Installation Steps

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/DisGuiser9/RecSys.git
   cd https://github.com/DisGuiser9/RecSys.git
   ```

2. Create and activate a virtual environment:
   ```bash
   conda create -n comp7240
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the project:
   ```bash
   python app.py
   ```

## Usage Instructions

- After starting the application, visit `http://localhost:5000`.
- Users can enter their user ID to receive personalized recommendations.
- The system offers various recommendation modes, including quick and accurate recommendations.

## File Structure

- `app.py`: The main file for the Flask application, containing routes and recommendation logic.
- `CF.py`: Module implementing the collaborative filtering algorithm.
- `dataset/`: Contains dataset files such as `Books.csv` and `Users.csv`.
- `templates/`: Contains HTML template files.
- `static/`: Contains static files like CSS and JavaScript.

## Contribution Guidelines

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Submit a Pull Request.

## License Information

This project is licensed under the MIT License. For more information, see the LICENSE file.