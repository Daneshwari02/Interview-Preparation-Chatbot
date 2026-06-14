from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import re
import json
import uuid
import PyPDF2
import fitz
import base64
from groq import Groq
from dotenv import load_dotenv
import pymongo
import certifi
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "mca_project_secret_key_123")

# Configurations
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY and not GROQ_API_KEY.startswith("your_"):
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None

# Initialize MongoDB Connection
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://resume_user:Resume%4012345@cluster0.lbv6d2x.mongodb.net/resume_analyzer_mca?retryWrites=true&w=majority"
)
try:
    mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    mongo_client.admin.command("ping")
    db = mongo_client["resume_analyzer_mca"]
    candidates_collection = db["candidates"]
    print("[SUCCESS] MongoDB Connected Successfully")
except Exception as e:
    print("[ERROR] MongoDB Connection Failed:", e)
    mongo_client = None
    db = None
    candidates_collection = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from pdfminer.high_level import extract_text

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file using PyMuPDF (fitz) for superior layout and embedded font handling."""
    text = ""
    if filepath.lower().endswith('.pdf'):
        try:
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
        except Exception as e:
            print(f"Error extracting text with PyMuPDF: {e}")
    return text.strip()

def extract_text_with_vision(filepath):
    """Uses Groq Vision to extract text from an image-based PDF (like Canva resumes or photos)."""
    if not client:
        return ""
    try:
        # Convert first page of PDF to image using PyMuPDF
        doc = fitz.open(filepath)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Higher resolution
        img_bytes = pix.tobytes("jpeg")
        doc.close()
        
        encoded_string = base64.b64encode(img_bytes).decode('utf-8')
        
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all the text from this resume image exactly as it appears. Do not add any extra commentary or markdown formatting, just return the raw text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
                    ]
                }
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in Vision extraction: {e}")
        return ""

def is_document_resume(text):
    """Uses Groq to determine if the document is a resume. Highly permissive to allow resumes with photos or formatting issues."""
    if not client:
        # Fallback heuristic
        text_lower = text.lower()
        resume_keywords = ['experience', 'education', 'skills', 'employment', 'projects', 'summary', 'profile', 'objective', 'contact']
        matches = sum(1 for kw in resume_keywords if kw in text_lower)
        return matches >= 1 # Very permissive

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a permissive document classifier. Determine if the provided text could potentially be a resume or CV. Even if the text contains garbage characters from photos, is very short, or is poorly formatted, if there is ANY indication it belongs to a job applicant (e.g., mentions of education, skills, work, or contains contact info), classify it as a resume. Return ONLY a JSON object with a boolean key 'is_resume'."
                },
                {
                    "role": "user",
                    "content": f"Text:\n{text[:2000]}"
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        data = json.loads(response.choices[0].message.content)
        return data.get("is_resume", True)
    except Exception as e:
        print(f"Error in is_document_resume: {e}")
        return True # Default to True on error to not block legitimate uploads

def calculate_ats_score(text):
    """Calculates ATS score based on sections, contact info, and content quality."""
    score = 0
    feedback = []
    text_lower = text.lower()
    
    # 1. Critical Sections Check (Weight: 60%)
    # These are non-negotiable for a professional resume
    sections = {
        "skills": ["skills", "technical skills", "technologies", "expertise", "core competencies"],
        "education": ["education", "academic profile", "qualification", "academic background"],
        "experience": ["experience", "work history", "professional experience", "employment history", "projects"]
    }
    
    sections_found = 0
    for section, keywords in sections.items():
        if any(key in text_lower for key in keywords):
            score += 20
            sections_found += 1
        else:
            feedback.append(f"Missing critical section: {section.capitalize()}")

    # 2. Contact Information (Weight: 20%)
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    
    has_email = bool(re.search(email_pattern, text))
    has_phone = bool(re.search(phone_pattern, text))

    if has_email: score += 10
    else: feedback.append("Email address missing.")
    
    if has_phone: score += 10
    else: feedback.append("Phone number missing.")

    # 3. Structural & Content Integrity (Weight: 20%)
    # - Length check
    word_count = len(text.split())
    if 200 <= word_count <= 1500:
        score += 10
    elif word_count < 100:
        score -= 20 # Penalty for extremely thin content
        feedback.append("Content is too brief for analysis.")
    
    # - Formatting check (simple line density check)
    lines = [line for line in text.split('\n') if line.strip()]
    if len(lines) > 15:
        score += 10
    else:
        feedback.append("Resume lacks standard structural depth (too few lines).")

    # Final logic for "ATS Valid"
    # A resume must have:
    # - At least 2 critical sections
    # - At least one contact method (email or phone)
    # - A minimum score of 50
    is_ats_valid = (sections_found >= 2) and (has_email or has_phone) and (score >= 50)
    
    if is_ats_valid and not feedback:
        feedback.append("Excellent! Your resume follows a professional structure.")
        
    return score, feedback, is_ats_valid

def get_ai_insights(text):
    """Uses Groq to extract skills and provide a profile summary in JSON format."""
    if not client:
        return ["AI analysis unavailable"], "AI Model not configured."

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional resume parser. Extract insights and return ONLY a JSON object with keys 'skills' (list of strings) and 'summary' (3-sentence string)."
                },
                {
                    "role": "user",
                    "content": f"Resume Text:\n{text[:4000]}"
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        skills = data.get("skills", [])
        summary = data.get("summary", "Professional analysis complete.")
        
        return (skills if skills else ["Check resume content"]), summary
    except Exception as e:
        print(f"Error in AI Insights: {e}")
        return ["Extraction failed"], f"Analysis failed: {str(e)}"

def get_detailed_analysis(text):
    """Uses Groq to perform a deep ATS audit and return structured metrics."""
    if not client:
        return None

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Career Coach and Resume Evaluator. Analyze the provided resume text and output a detailed report in JSON format. CRITICAL INSTRUCTION: Do NOT evaluate the ATS compatibility, machine readability, or formatting artifacts of the text. Assume the text extraction is fine. Your ONLY job is to evaluate the QUALITY, IMPACT, and STRENGTH of the applicant's ACTUAL CAREER CONTENT (their skills, experience, achievements, education). Assign a fair, realistic 'score' (0-100) for each field based on how impressive the content is to a human hiring manager. Never assign 0% unless the section is completely missing. The JSON must contain: 'overall_score' (0-100), 'overall_summary' (a 2-3 sentence paragraph explaining the overall score, strengths, and major ATS gaps), and a 'fields' array. Each object in the 'fields' array must have 'name' (e.g., 'Formatting', 'Skills', 'Experience', 'Education', 'Projects', 'Quantification'), 'score' (number 0-100), and 'improvements' (list of 1-3 actionable issues found). Each improvement in the list must be an object with four keys: 'severity' (string: 'Critical', 'Moderate', or 'Minor'), 'issue_title' (string: short bold title of the issue), 'description' (string: detailed explanation of the ATS red flag or issue context), and 'suggestion' (string: specific actionable advice to fix it with an example)."
                },
                {
                    "role": "user",
                    "content": f"Resume Text:\n{text[:4000]}"
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in Detailed Analysis: {e}")
        return None

def generate_interview_prep(skills, text):
    """Generates 15 structured interview questions tailored to the resume using Groq JSON mode."""
    if not client:
        return {"Easy": [], "Medium": [], "Hard": []}

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert interviewer. Generate 15 customized interview questions (5 Easy, 5 Medium, 5 Hard) based on the skills and resume context. For each question, provide a concise 'suggested_answer' that highlights how the candidate should frame their response based on their resume content. Return ONLY a JSON object with keys 'Easy', 'Medium', and 'Hard', each containing a list of objects with 'question' and 'suggested_answer' keys."
                },
                {
                    "role": "user",
                    "content": f"Skills: {', '.join(skills)}\n\nResume Context: {text[:2000]}"
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in Q&A generation: {e}")
        return {"Easy": [], "Medium": [], "Hard": []}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Consolidated error message for ALL non-ATS or failed cases
    GENERIC_ERROR = "Invalid file or not a PDF. Please ensure you upload a valid PDF document."

    if 'resume' not in request.files:
        return render_template('error.html', error_message=GENERIC_ERROR)

    file = request.files['resume']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('error.html', error_message=GENERIC_ERROR)

    # Use UUID to ensure unique filename and avoid collision/caching issues
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        file.save(filepath)

        # 1. Extract Text
        text = extract_text_from_pdf(filepath)
        is_image_based = False
        
        # If normal text extraction fails or gets very little text, try vision!
        if not text or len(text.strip()) < 50:
            vision_text = extract_text_with_vision(filepath)
            if vision_text and len(vision_text.strip()) > 50:
                text = vision_text
            else:
                is_image_based = True
                text = "The uploaded PDF appears to be an image or scanned photo. No machine-readable text was found. ATS systems cannot read image-based resumes."

        # 1.5 Check if Document is a Resume (Skip check if it's purely an image, as we can't read it anyway)
        if not is_image_based:
            if not is_document_resume(text):
                if os.path.exists(filepath):
                    os.remove(filepath)
                # Ensure the user gets the exact message "not a resume try again"
                return render_template('error.html', error_message="not a resume try again")

        # 2. ATS Validation Scoring
        ats_score, ats_feedback, is_ats_valid = calculate_ats_score(text)
        
        # 3. AI Insights (Skills & Summary) - FRESH CALL
        skills, summary = get_ai_insights(text)

        # 4. Interview Prep (Q&A)
        interview_data = generate_interview_prep(skills, text)

        # 5. NEW: Detailed ATS Analysis
        detailed_analysis = get_detailed_analysis(text)
        
        # 6. Save to MongoDB
        candidate_data = {
            "filename": file.filename,
            "resume_text": text,
            "skills": skills,
            "summary": summary,
            "ats_score": detailed_analysis.get("overall_score", ats_score) if detailed_analysis else ats_score,
            "ats_overall_summary": detailed_analysis.get("overall_summary", "Review the detailed breakdown below.") if detailed_analysis else "Review the detailed breakdown below.",
            "ats_feedback": detailed_analysis.get("improvements", ats_feedback) if detailed_analysis else ats_feedback,
            "ats_breakdown": detailed_analysis.get("breakdown", {}) if detailed_analysis else {},
            "ats_fields": detailed_analysis.get("fields", []) if detailed_analysis else {},
            "interview_prep": interview_data,
            "user_id": None,
            "is_guest": True
        }
        result = candidates_collection.insert_one(candidate_data)
        candidate_id = str(result.inserted_id)

        # Cleanup file immediately after processing
        if os.path.exists(filepath):
            os.remove(filepath)

        # Post-Redirect-Get pattern: Redirect to the specific candidate results
        return redirect(url_for('view_candidate', candidate_id=candidate_id))

    except Exception as e:
        print(f"Upload Error: {e}")
        if os.path.exists(filepath): os.remove(filepath)
        return render_template('error.html', error_message=GENERIC_ERROR)

@app.route('/candidate/<candidate_id>')
def view_candidate(candidate_id):
    try:
        candidate = candidates_collection.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            return render_template('error.html', error_message="Candidate not found.")
        
        return render_template(
            "results.html",
            skills=candidate.get("skills", []),
            summary=candidate.get("summary", ""),
            ats_score=candidate.get("ats_score", 0),
            ats_overall_summary=candidate.get("ats_overall_summary", "Review the detailed breakdown below."),
            ats_feedback=candidate.get("ats_feedback", []),
            ats_breakdown=candidate.get("ats_breakdown", {}),
            ats_fields=candidate.get("ats_fields", []),
            interview_data=candidate.get("interview_prep", {}),
            jd_match=candidate.get("jd_match", None),
            target_jd=candidate.get("target_jd", ""),
            candidate_id=str(candidate["_id"]),
            filename=candidate.get("filename", "Unknown")
        )
    except Exception as e:
        return render_template('error.html', error_message=f"Error retrieving candidate: {str(e)}")

@app.route('/api/match_jd', methods=['POST'])
def match_jd():
    data = request.json
    candidate_id = data.get('candidate_id')
    job_description = data.get('job_description')

    if not candidate_id or not job_description:
        return jsonify({"error": "Missing candidate_id or job_description"}), 400

    if not client:
        return jsonify({"error": "AI Model not configured"}), 500

    try:
        candidate = candidates_collection.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            return jsonify({"error": "Candidate not found"}), 404

        resume_text = candidate.get("resume_text", "")
        if not resume_text:
            return jsonify({"error": "Resume text not available"}), 400

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ATS and recruitment AI. Compare the applicant's resume against the provided Job Description. Return ONLY a JSON object with: 'match_percentage' (integer 0-100), 'missing_skills' (list of strings, up to 5 critical skills missing from resume but required in JD), and 'match_analysis' (a 2-sentence summary explaining the fit)."
                },
                {
                    "role": "user",
                    "content": f"Resume Text:\n{resume_text[:3000]}\n\nJob Description:\n{job_description[:3000]}"
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Save to DB
        candidates_collection.update_one(
            {"_id": ObjectId(candidate_id)},
            {"$set": {"jd_match": result, "target_jd": job_description}}
        )

        return jsonify(result)
    except Exception as e:
        print(f"Error in match_jd: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/fix_resume_section', methods=['POST'])
def fix_resume_section():
    data = request.json
    candidate_id = data.get('candidate_id')
    issue_context = data.get('issue_context')

    if not candidate_id or not issue_context:
        return jsonify({"error": "Missing candidate_id or issue_context"}), 400

    if not client:
        return jsonify({"error": "AI Model not configured"}), 500

    try:
        candidate = candidates_collection.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            return jsonify({"error": "Candidate not found"}), 404

        resume_text = candidate.get("resume_text", "")

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Resume Writer. Your task is to rewrite a specific section or bullet point of the provided resume to fix an issue. Use the metric-driven XYZ format ('Accomplished [X] as measured by [Y], by doing [Z]') where applicable. Be concise, professional, and impactful. Output ONLY the rewritten text, nothing else."
                },
                {
                    "role": "user",
                    "content": f"Resume Text:\n{resume_text[:3000]}\n\nIssue to Fix / Section to Rewrite:\n{issue_context}"
                }
            ],
            model="llama-3.3-70b-versatile"
        )
        
        rewritten_text = response.choices[0].message.content.strip()
        return jsonify({"rewritten_text": rewritten_text})
    except Exception as e:
        print(f"Error in fix_resume_section: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    candidate_id = data.get('candidate_id')
    messages = data.get('messages', [])

    if not candidate_id or not messages:
        return jsonify({"error": "Missing candidate_id or messages"}), 400

    if not client:
        return jsonify({"error": "AI Model not configured"}), 500

    try:
        candidate = candidates_collection.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            return jsonify({"error": "Candidate not found"}), 404

        resume_text = candidate.get("resume_text", "")
        
        system_msg = {
            "role": "system",
            "content": f"You are a helpful, expert Resume Career Coach. You are assisting the user in improving their resume. Base your advice entirely on the context of their current resume. Keep answers concise and highly actionable. \n\nCandidate's Resume:\n{resume_text[:3000]}"
        }
        
        api_messages = [system_msg] + messages

        response = client.chat.completions.create(
            messages=api_messages,
            model="llama-3.3-70b-versatile"
        )
        
        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin-vault')
def admin_vault():
    try:
        # Fetch all candidates, sorted by most recent first
        all_candidates = list(candidates_collection.find().sort("_id", -1))
        return render_template('admin_vault.html', candidates=all_candidates)
    except Exception as e:
        return render_template('error.html', error_message=f"Database error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)