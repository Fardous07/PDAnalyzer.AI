# PDAnalyzer.AI 🎙️📊

**AI-Powered Political Discourse Analysis Platform**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-18.x-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

---

## Project Info

### Principal / Supervisor
**Fazle Rabbi**  
**Associate Professor**  
**Department of Information and Media Studies**

### Program / Initiative
**SFI Innovation**

### Project Purpose
This project is developed as part of an SFI Innovation effort to build a practical, research-aligned system for political discourse analysis: uploading speeches, extracting structured evidence, assigning ideology families, and supporting comparative analysis through interactive dashboards.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Ideology System](#ideology-system)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Guide](#setup-guide)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Environment Configuration](#environment-configuration)
  - [Database Setup](#database-setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Analysis Pipeline](#analysis-pipeline)
- [User Interface](#user-interface)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Project Information](#project-information)
- [License](#license)

---

## Overview

**PDAnalyzer.AI** is a comprehensive full-stack application designed for analyzing political speeches and discourse using advanced artificial intelligence technologies. The platform provides evidence-based ideological classification, semantic analysis, and powerful comparison tools for understanding political rhetoric.

This system focuses on:
- **Evidence-based classification** with quantifiable metrics (counts, shares, confidence scores)
- **Consistent ideology families** using a 4-family classification system
- **Comparative analysis** with similarity matrices and 2D ideological mapping
- **Project-based workflow** where each speech is managed as a discrete project

---

## Core Features

### Speech & Project Management
- Create projects by uploading speeches in multiple formats (text, PDF, DOCX, audio, video)
- Store and manage multiple speeches per user account
- Rename, delete, and refresh projects with intuitive controls
- Efficient batch operations for streamlined workflow

### Transcription Support (Optional)
- AI-powered transcription for audio and video uploads
- Automatic text extraction from media files when enabled
- Integration with advanced speech-to-text services

### Ideology & Evidence Extraction
- Extract structured evidence units from comprehensive speech analysis
- Assign ideology family to each unit with confidence scoring
- Compute comprehensive metrics including:
  - Total evidence units counted
  - Family distribution shares (percentage breakdown)
  - Average confidence scores across analysis
  - MARPOR-style coding integration for political science research

### Comparison Dashboard
- Select and compare multiple speeches (up to 8 simultaneously)
- Compare ideology distribution across different speeches
- Generate similarity matrix from semantic vectors
- Render 2D ideology map with economic and social axes
- Visual insights for quick understanding of political positioning

### Authentication & Security
- JWT-based authentication for secure user access
- Complete user registration and login system
- Protected routes with AuthGuard implementation
- Secure session management with token refresh capability

---

## Ideology System

PDAnalyzer.AI employs a sophisticated **4-family ideology classification** system for analyzing political discourse:

**Four Ideology Families:**

1. **Libertarian** - Emphasis on individual freedom and minimal government intervention
2. **Authoritarian** - Strong government control and traditional values
3. **Economic-Left** - Social welfare, wealth redistribution, and workers' rights
4. **Economic-Right** - Free market economy, private enterprise, and fiscal conservatism

**2D Ideological Mapping:**

The system uses a two-dimensional political compass for nuanced analysis:
- **X-axis (Economic)**: Left (social welfare) ↔ Right (free market)
- **Y-axis (Social)**: Authoritarian (control) ↔ Libertarian (freedom)

```
                    Libertarian 🕊️
                    (Freedom)
                         |
                         |
            Libertarian  |  Libertarian
            Left         |  Right
                         |
                         |
    Economic ────────────┼──────────── Economic
    Left 🌹              |              Right 💼
    (Welfare)            |              (Free Market)
                         |
                         |
            Authoritarian|  Authoritarian
            Left         |  Right
                         |
                         |
                    Authoritarian 🏛️
                    (Control)


    Quadrant Descriptions:
    
    ┌─────────────────────┬─────────────────────┐
    │  Libertarian-Left   │  Libertarian-Right  │
    │  • Social freedom   │  • Individual       │
    │  • Economic equity  │    liberty          │
    │  • Progressive      │  • Free market      │
    │    values           │  • Minimal govt     │
    ├─────────────────────┼─────────────────────┤
    │  Authoritarian-Left │  Authoritarian-     │
    │  • State control    │    Right            │
    │  • Collective       │  • Traditional      │
    │    ownership        │    values           │
    │  • Central planning │  • Strong authority │
    └─────────────────────┴─────────────────────┘
```

This dual-axis approach provides more sophisticated understanding of political positioning beyond traditional left-right spectrum analysis. Each speech is plotted on this 2D map based on:
- **Economic score**: Calculated from fiscal policy, market regulation, and welfare statements
- **Social score**: Derived from civil liberties, government authority, and social policy positions

---

## Architecture

The application follows a modern three-tier architecture:

```
┌─────────────────────────────────────────────┐
│        React Frontend (Vite)                │
│  • Modern UI Components                     │
│  • Real-time Updates                        │
│  • Responsive Design                        │
└──────────────────┬──────────────────────────┘
                   │
                   │ REST API (Axios)
                   │ JSON over HTTPS
                   ↓
┌─────────────────────────────────────────────┐
│        FastAPI Backend                      │
│  • Speech Processing Engine                 │
│  • Analysis Orchestration                   │
│  • JWT Authentication                       │
└──────────────────┬──────────────────────────┘
                   │
                   │ SQLAlchemy ORM
                   ↓
┌─────────────────────────────────────────────┐
│        PostgreSQL Database                  │
│  • Users & Authentication                   │
│  • Speeches & Projects                      │
│  • Analysis Results                         │
│  • Evidence Units                           │
└──────────────────┬──────────────────────────┘
                   │
                   ├──→ OpenAI / Anthropic
                   └──→ Whisper (Optional)
```

**Architecture Components:**

1. **Presentation Layer** - React frontend with modern UI/UX
2. **API Layer** - FastAPI RESTful endpoints with validation
3. **Business Logic Layer** - Analysis engine and service orchestration
4. **Data Layer** - PostgreSQL with SQLAlchemy ORM
5. **External Services** - Optional LLM providers and transcription services

---

## Tech Stack

### Backend Technologies

- **FastAPI** - Modern Python web framework for building APIs
- **SQLAlchemy** - SQL toolkit and Object-Relational Mapping (ORM)
- **PostgreSQL** - Advanced open-source relational database
- **JWT** - JSON Web Tokens for secure authentication
- **OpenAI API** - GPT models and Whisper transcription (optional)
- **Anthropic API** - Claude models for analysis (optional)

### Frontend Technologies

- **React 18.x** - Modern JavaScript library for building user interfaces
- **Vite** - Next-generation frontend build tool
- **React Router v6** - Declarative routing for React applications
- **Axios** - Promise-based HTTP client for API communication
- **Context API** - Built-in React state management
- **Tailwind CSS** - Utility-first CSS framework for styling

---

## Project Structure

```
PDAnalyzer.AI/
│
├── backend/                          # Backend application
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/              # API endpoint definitions
│   │   │       ├── analysis.py      # Analysis operations
│   │   │       ├── auth.py          # Authentication endpoints
│   │   │       ├── speeches.py      # Speech CRUD operations
│   │   │       └── users.py         # User management
│   │   │
│   │   ├── database/
│   │   │   ├── connection.py        # Database connection setup
│   │   │   └── models.py            # SQLAlchemy data models
│   │   │
│   │   ├── services/
│   │   │   ├── speech_ingestion.py  # Main analysis engine
│   │   │   ├── transcription.py     # Audio/video processing
│   │   │   ├── ideology_scoring.py  # Ideology classification logic
│   │   │   ├── marpor_definitions.py # MARPOR framework
│   │   │   ├── question_generator.py # AI question generation
│   │   │   ├── llm_router.py        # LLM provider routing
│   │   │   └── attribution_parser.py # Response parsing utilities
│   │   │
│   │   ├── config.py                # Application configuration
│   │   └── main.py                  # FastAPI application entry
│   │
│   ├── requirements.txt             # Python package dependencies
│   └── .env                         # Environment variables
│
├── frontend/                         # Frontend application
│   ├── src/
│   │   ├── components/              # Reusable React components
│   │   ├── context/                 # React Context providers
│   │   ├── pages/                   # Page-level components
│   │   ├── services/                # API service layer
│   │   ├── App.jsx                  # Main application component
│   │   └── main.jsx                 # Application entry point
│   │
│   ├── package.json                 # Node.js dependencies
│   └── .env                         # Frontend configuration
│
├── .gitignore                       # Git ignore patterns
├── README.md                        # Project documentation
└── LICENSE                          # MIT License file
```

---

## Setup Guide

### Prerequisites

Ensure the following software is installed on your system:

- **Python 3.9 or higher** - [Download Python](https://www.python.org/downloads/)
- **Node.js 16 or higher** (includes npm) - [Download Node.js](https://nodejs.org/)
- **PostgreSQL 13 or higher** - [Download PostgreSQL](https://www.postgresql.org/download/)
- **Git** - [Download Git](https://git-scm.com/downloads)

**Optional Requirements:**
- OpenAI API Key (for GPT-4 analysis and Whisper transcription)
- Anthropic API Key (for Claude model integration)

---

### Backend Setup

**Step 1: Navigate to Backend Directory**

```bash
cd backend
```

**Step 2: Create Python Virtual Environment**

```bash
python -m venv investigeenv
```

**Step 3: Activate Virtual Environment**

For Windows:
```bash
investigeenv\Scripts\activate
```

For macOS/Linux:
```bash
source investigeenv/bin/activate
```

**Step 4: Install Python Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Frontend Setup

**Step 1: Navigate to Frontend Directory**

```bash
cd frontend
```

**Step 2: Install Node Dependencies**

```bash
npm install
```

---

### Environment Configuration

#### Backend Configuration

Create a file named `.env` in the `backend` directory with the following configuration:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/pdanalyzer_db

# JWT Authentication Settings
SECRET_KEY=change-me-to-a-secure-random-string
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI/ML API Keys (Optional)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# CORS Configuration
CORS_ORIGINS=http://localhost:5173

# Application Settings
ENVIRONMENT=development
DEBUG=True
```

**Important Notes:**
- Replace `username` and `password` with your PostgreSQL credentials
- Generate a secure random string for `SECRET_KEY`
- Add your API keys if using AI features

#### Frontend Configuration

Create a file named `.env` in the `frontend` directory:

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

### Database Setup

**Step 1: Create PostgreSQL Database**

Open PostgreSQL command line (psql):

```bash
psql -U postgres
```

Execute the following SQL command:

```sql
CREATE DATABASE pdanalyzer_db;
```

Exit psql:

```sql
\q
```

**Step 2: Verify Database Connection**

Ensure your `backend/.env` file's `DATABASE_URL` matches your PostgreSQL setup.

**Step 3: Initialize Database Tables**

Tables will be automatically created when you first run the backend application. The system uses SQLAlchemy's automatic table creation feature.

---

## Running the Application

### Start Backend Server

**Step 1: Navigate to Backend Directory**

```bash
cd backend
```

**Step 2: Activate Virtual Environment**

For Windows:
```bash
investigeenv\Scripts\activate
```

For macOS/Linux:
```bash
source investigeenv/bin/activate
```

**Step 3: Start the FastAPI Server**

```bash
uvicorn app.main:app --reload --port 8000
```

The backend will be running at:
- **Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **OpenAPI Specification**: http://localhost:8000/openapi.json

---

### Start Frontend Development Server

Open a new terminal window and execute:

**Step 1: Navigate to Frontend Directory**

```bash
cd frontend
```

**Step 2: Start Development Server**

```bash
npm run dev
```

The frontend will be accessible at:
- **Application**: http://localhost:5173

---

## API Documentation

PDAnalyzer.AI provides comprehensive API documentation through FastAPI's automatic documentation generation.

### Accessing API Documentation

**Swagger UI (Interactive Documentation)**
- URL: http://localhost:8000/docs
- Features:
  - Interactive API testing interface
  - Try endpoints directly in the browser
  - View detailed request/response schemas
  - Test authentication flows

**OpenAPI JSON Specification**
- URL: http://localhost:8000/openapi.json
- Use for:
  - Importing into API clients (Postman, Insomnia)
  - Generating client SDKs
  - Integration with other tools

### API Endpoint Categories

**Authentication Endpoints** (`/auth/*`)
- User registration
- Login and token generation
- Token refresh
- Logout

**User Management** (`/users/*`)
- Get user profile
- Update user information
- View user statistics
- Manage user settings

**Speech Operations** (`/speeches/*`)
- Create new speeches (upload)
- Retrieve speech details
- Update speech information
- Delete speeches
- Trigger analysis

**Analysis Operations** (`/analysis/*`)
- Get analysis results
- Compare multiple speeches
- Generate similarity matrices
- Access 2D ideology mapping

---

## Analysis Pipeline

The speech analysis workflow consists of seven sequential stages:

**Stage 1: Input Reception**
- Accept text input, PDF files, DOCX documents, or media files
- Validate file format and size
- Queue for processing

**Stage 2: Text Preprocessing**
- Clean and normalize input text
- Remove special characters and formatting
- Validate text length and quality

**Stage 3: Transcription (Optional)**
- Process audio/video files through Whisper API
- Convert speech to text
- Handle multiple audio formats

**Stage 4: Segmentation**
- Split text into analyzable evidence units
- Identify key statements and segments
- Create contextual chunks for analysis

**Stage 5: Classification**
- Assign ideology family to each evidence unit
- Apply MARPOR coding framework
- Calculate confidence scores for classifications
- Extract supporting evidence

**Stage 6: Aggregation**
- Calculate total evidence units
- Compute ideology family distribution (percentages)
- Calculate average confidence scores
- Generate 2D ideology coordinates (economic and social axes)

**Stage 7: Storage & Retrieval**
- Save complete analysis results to database
- Make results available for UI display
- Generate comparison metadata for dashboard

**Pipeline Characteristics:**
- Asynchronous background processing for long-running analyses
- Real-time progress tracking and status updates
- Persistent storage of all results
- Error handling and recovery mechanisms

---

## User Interface

### Application Screens

**1. Authentication Pages**
- User registration with validation
- Secure login interface
- JWT token management
- Password encryption and security

**2. Project Management**
- Create new projects with speech uploads
- Support for multiple file formats (text, PDF, DOCX, audio, video)
- Configure analysis parameters
- View project list and status

**3. Analysis Page**

*Speech Details Section:*
- Display full speech text
- Show metadata (speaker, date, location, context)
- Word count and basic statistics

*Evidence Units Display:*
- List all extracted evidence segments
- Show ideology label for each unit
- Display confidence scores
- Link to MARPOR code assignments

*Ideology Classification Results:*
- Primary ideology family identification
- Distribution chart across all four families
- Confidence metrics and scores
- 2D political compass position visualization

**4. Comparison Dashboard**

*Speech Selection Interface:*
- Choose multiple speeches (maximum 8)
- Filter by date, speaker, or ideology
- Bulk selection tools

*Visualization Components:*
- Side-by-side family share comparison (bar charts)
- Similarity matrix showing relationships between speeches
- 2D evidence map plotting all speeches on political compass
- Statistical comparison tables

---

## Troubleshooting

### Common Issues and Solutions

**Issue: Backend Server Not Reachable**

Problem: Frontend cannot establish connection with backend

Solutions:
1. Verify backend is running on port 8000:
   ```bash
   cd backend
   investigeenv\Scripts\activate  # Windows
   source investigeenv/bin/activate  # macOS/Linux
   uvicorn app.main:app --reload --port 8000
   ```

2. Check frontend `.env` configuration:
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```

3. Restart both frontend and backend servers

---

**Issue: CORS (Cross-Origin Resource Sharing) Errors**

Problem: Browser blocks requests due to CORS policy

Solutions:
1. Verify backend `.env` includes frontend URL:
   ```env
   CORS_ORIGINS=http://localhost:5173
   ```

2. Restart backend server after modifying `.env`

3. Clear browser cache and reload page

---

**Issue: Authentication/Token Problems**

Problem: Login fails or authentication errors occur

Solutions:
1. Clear browser localStorage:
   - Open browser developer console (F12)
   - Go to Application/Storage tab
   - Clear localStorage
   - Alternatively, run: `localStorage.clear()` in console

2. Re-login to obtain fresh authentication token

3. Verify backend `/auth/login` endpoint is operational:
   ```bash
   curl -X POST http://localhost:8000/auth/login \
     -d "username=testuser&password=testpass"
   ```

4. Ensure `SECRET_KEY` is properly set in backend `.env`

---

**Issue: Database Connection Failure**

Problem: Cannot establish connection to PostgreSQL

Solutions:
1. Verify PostgreSQL service is running:
   - Windows: Check Services application
   - macOS: Run `brew services list`
   - Linux: Run `systemctl status postgresql`

2. Test database connection manually:
   ```bash
   psql -U postgres -d pdanalyzer_db
   ```

3. Verify `DATABASE_URL` in backend `.env` has correct credentials

4. Ensure database `pdanalyzer_db` exists

---

**Issue: Python Module Import Errors**

Problem: "ModuleNotFoundError" when running backend

Solutions:
1. Confirm virtual environment is activated:
   ```bash
   cd backend
   investigeenv\Scripts\activate  # Windows
   source investigeenv/bin/activate  # macOS/Linux
   ```

2. Reinstall all requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify Python version is 3.9 or higher:
   ```bash
   python --version
   ```

---

**Issue: npm Install Failures**

Problem: Frontend dependency installation errors

Solutions:
1. Clear npm cache:
   ```bash
   npm cache clean --force
   ```

2. Delete existing modules and reinstall:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

3. Try clean install:
   ```bash
   npm ci
   ```

---

## Contributing

Contributions to PDAnalyzer.AI are welcome and appreciated. To contribute:

**Step 1: Fork the Repository**

Visit the GitHub repository and click the "Fork" button.

**Step 2: Clone Your Fork**

```bash
git clone https://github.com/YOUR_USERNAME/PDAnalyzer.AI.git
cd PDAnalyzer.AI
```

**Step 3: Create Feature Branch**

```bash
git checkout -b feature/your-feature-name
```

**Step 4: Make Your Changes**

- Write clean, well-documented code
- Follow existing code style and conventions
- Add tests for new features when applicable

**Step 5: Commit Changes**

```bash
git add .
git commit -m "Add descriptive commit message"
```

**Step 6: Push to Your Fork**

```bash
git push origin feature/your-feature-name
```

**Step 7: Create Pull Request**

- Navigate to the original repository
- Click "New Pull Request"
- Provide clear description of changes

**Contribution Guidelines:**
- Follow PEP 8 style guide for Python code
- Use ESLint conventions for JavaScript/React code
- Write meaningful commit messages
- Update documentation for new features
- Ensure all existing tests pass

---

## Project Information

### Academic Context

This project was developed as part of academic research in political discourse analysis and artificial intelligence applications.

**Author:**
- **Name**: **Fardous Hasan**
- **GitHub**: [Fardous07](https://github.com/Fardous07)
- **Email**: fardous.amath@gmail.com

**Academic Supervision:**
- **Supervisor**: **Fazle Rabbi**
- **Department**: **Associate Professor**
- **Institution**: **Department of Information and Media Studies**
  

**Project Details:**
- **Project Type**: Research Project 
- **Field**: Political Science, Artificial Intelligence, MARPOR code
- **Year**: 2024

**Repository:**
- **GitHub**: https://github.com/Fardous07/PDAnalyzer.AI

---

## Acknowledgments

Special thanks to:
- **MARPOR Project** for providing the political content analysis framework
- **OpenAI** for GPT-4 and Whisper API access
- **Anthropic** for Claude model integration
- **FastAPI Community** for excellent documentation and support
- **React Community** for powerful tools and libraries
- **Academic Supervisor** for guidance and support throughout the project
- **Department Faculty** for valuable feedback and insights

---

## Contact & Support

For questions, issues, or support:

- **GitHub Issues**: [Report Issues](https://github.com/Fardous07/PDAnalyzer.AI/issues)
- **Email**: fardous.amath@gmail.com
- **Documentation**: Available in repository

---

**⭐ If you find this project useful, please consider starring the repository!**

---

*Last Updated: February 2024*
