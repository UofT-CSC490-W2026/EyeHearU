# ASL Learning Platform: Project Outline & Strategy

## Executive Summary

**Project Name Ideas:** SignSense / SignFlow / HandsOn ASL / SignCoach

**One-liner:** An AI-powered ASL learning platform that provides real-time feedback on sign accuracy using computer vision, making ASL education accessible, personalized, and effective.

**Why this stands out:**
- Combines computer vision, pose estimation, and ML classification
- Clear social impact (accessibility, inclusion)
- Real user base with acquisition potential
- Full-stack complexity (React/Next.js + Python ML backend + model training)
- B2C and B2B potential (individuals + schools/employers)

---

# PART ONE: Interest Statements (20 marks)

## Team Problem Statement

Our team is passionate about bridging communication gaps between the Deaf and hearing communities. Despite ASL being the third most used language in the United States, learning resources remain limited, expensive, and often lack the interactive feedback necessary for mastering a visual-gestural language. Traditional ASL courses cost $200-500 and require in-person attendance, creating significant barriers to access. Online resources exist but provide no feedback on whether learners are signing correctly—a critical gap since ASL requires precise hand shapes, movements, and facial expressions.

We care about this problem because accessibility should not be a privilege. Over 500,000 people in the US use ASL as their primary language, yet most hearing people have no practical way to learn it. By creating an AI-powered learning platform with real-time feedback, we can democratize ASL education, foster inclusion, and create meaningful connections between communities. As a team of engineers, we're excited to apply cutting-edge ML techniques (pose estimation, gesture recognition, sequence modeling) to a problem with genuine human impact.

## Individual Interest Statements

**[Team Member 1 Name]:** I'm excited to lead the computer vision pipeline, implementing MediaPipe pose estimation and training custom gesture classification models—this intersection of CV and real-world impact is exactly what I want to build my career around.

**[Team Member 2 Name]:** I'm passionate about building the full-stack application, creating an intuitive and accessible UI/UX that makes learning ASL feel like a game rather than a chore.

**[Team Member 3 Name]:** I want to focus on the ML model training and evaluation, experimenting with different architectures (CNNs, LSTMs, Transformers) to achieve high accuracy on sign recognition.

**[Team Member 4 Name]:** I'm interested in the product and growth side—conducting user research, implementing analytics, and driving real user acquisition to validate our solution.

---

# PART TWO: Landscape Analysis (30 marks)

## Competitive Landscape Table

| # | Relevant Item | Type | Description | Commentary |
|---|---------------|------|-------------|------------|
| 1 | **SignAll** | Company | Enterprise ASL translation using depth cameras and ML. Raised $1.2M. Targets B2B (government, healthcare). | Gap: Requires expensive hardware, not accessible for individual learners. No learning/feedback focus. |
| 2 | **Lingvano** | Company | ASL learning app with video lessons and quizzes. 100K+ downloads. | Gap: No real-time feedback on user signs—just passive video watching. We can differentiate with AI feedback. |
| 3 | **SignSchool** | Company | Free ASL dictionary and courses with 10K+ signs. Web-based. | Gap: Reference tool, not interactive learning. No practice mode or feedback mechanism. |
| 4 | **The ASL App** | Company | Mobile app with video dictionary and learning mode. 500K+ downloads. | Gap: One-way learning (watch videos), no verification that user is signing correctly. |
| 5 | **Duolingo** | Company | Gamified language learning platform. 500M+ users. No ASL support. | Opportunity: Proven gamification model we can adapt. ASL is frequently requested but not offered due to CV complexity. |
| 6 | **MediaPipe** | Open Source | Google's ML framework for hand/pose tracking. Real-time, runs in browser. | Key enabler: Provides hand landmark detection we can build on. 21 hand landmarks at 30fps. |
| 7 | **OpenPose** | Open Source | CMU's body/hand pose estimation library. Research-grade accuracy. | Alternative to MediaPipe. Heavier but more accurate. Good for training data generation. |
| 8 | **"Word-level Deep Sign Language Recognition from Video" (Joze et al., 2019)** | Research | I3D + attention model for word-level ASL recognition. 83% accuracy on WLASL dataset. | Validates CNN+RNN approach. Shows word-level recognition is feasible with ~2000 sign vocabulary. |
| 9 | **"WLASL: A Large-Scale Dataset for Word-Level American Sign Language" (Li et al., 2020)** | Research | Largest ASL video dataset with 2000 words, 21K videos. Benchmark for ASL recognition. | Critical resource: Pre-existing labeled data we can use for training. Established benchmark to compare against. |
| 10 | **"Real-Time Sign Language Detection using Human Pose Estimation" (Mujahid et al., 2021)** | Research | MediaPipe + LSTM for real-time alphabet recognition. 95%+ accuracy on fingerspelling. | Validates our technical approach. Shows real-time is achievable with pose landmarks. |

## Key Insights from Landscape Analysis

1. **Gap in market:** No consumer-focused app provides real-time AI feedback on ASL signing
2. **Technical feasibility:** MediaPipe + ML models can achieve high accuracy in real-time
3. **Data availability:** WLASL and other datasets provide training data
4. **Monetization model:** Freemium (Duolingo-style) + B2B (schools, employers) is proven
5. **Differentiation opportunity:** Interactive feedback is our unique value proposition

---

# PART THREE: Project Outline (30 marks)

## Problem Statement

Learning American Sign Language is fundamentally different from learning spoken languages, yet existing digital learning tools treat it the same way—showing videos and quizzes without verifying if learners can actually produce signs correctly. This creates a critical feedback gap: learners have no way to know if their hand shapes, movements, or expressions are accurate until they interact with a fluent signer. The result is slow progress, bad habits, and frustrated learners who give up.

## Proposed Solution

**SignSense** is a web-based ASL learning platform that uses computer vision and machine learning to provide real-time feedback on sign accuracy. Users learn through structured lessons, practice with their webcam, and receive instant feedback on their signing—similar to how Duolingo teaches speaking through voice recognition, but for visual language.

**Core Features:**
1. **Structured curriculum:** Beginner to intermediate ASL lessons (alphabet, common phrases, vocabulary)
2. **Real-time sign checking:** Camera analyzes user's signs and provides accuracy feedback
3. **Practice mode:** Flashcard-style drilling with AI scoring
4. **Progress tracking:** XP, streaks, and skill trees (gamification)
5. **Sign dictionary:** Reference for 500+ signs with video examples

## High-Level Technical Approach

### Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend API   │────▶│   ML Service    │
│  (Next.js/React)│     │   (FastAPI)     │     │  (Python/ONNX)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   MediaPipe.js            PostgreSQL            Model Artifacts
   (Browser CV)            (User data)           (S3/Cloud)
```

### Tech Stack
- **Frontend:** Next.js 14, TypeScript, TailwindCSS, MediaPipe.js
- **Backend:** FastAPI (Python), PostgreSQL, Redis (caching)
- **ML:** PyTorch, MediaPipe, ONNX (for browser inference)
- **Infra:** Vercel (frontend), Railway/Fly.io (backend), AWS S3 (models)

### Milestones

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1-2 | Foundation | Project setup, basic UI, MediaPipe integration in browser |
| 3-4 | ML Pipeline | Data preprocessing, initial model training on WLASL subset |
| 5-6 | Core Features | Sign recognition API, feedback UI, 26 letters (alphabet) working |
| 7-8 | Expansion | 50+ word vocabulary, lesson structure, user accounts |
| 9-10 | Polish & Launch | Gamification, progress tracking, beta launch |
| 11-12 | Traction | User acquisition, feedback iteration, demo prep |

## Unknowns to Investigate

1. **Model accuracy vs. latency tradeoff:** Can we achieve >85% accuracy while maintaining real-time (<200ms) inference in the browser?
2. **Lighting/background robustness:** How do we handle varied webcam conditions?
3. **Motion sign recognition:** Static signs (letters) are easier; how do we handle signs with movement (words)?
4. **User feedback design:** What's the most effective way to show users *what* they're doing wrong?
5. **Dataset bias:** Is WLASL diverse enough, or do we need to collect additional training data?
6. **Browser ML performance:** Will ONNX.js/TensorFlow.js run smoothly on average hardware?

---

# PART FOUR: Press Release (20 marks)

## Press Release

**FOR IMMEDIATE RELEASE**

### SignSense Launches AI-Powered ASL Learning Platform, Making Sign Language Education Accessible to Everyone

*First platform to provide real-time feedback on sign language accuracy using computer vision*

**TORONTO, ON – [Date]** – SignSense today announced the launch of its AI-powered American Sign Language learning platform, bringing interactive, feedback-driven ASL education to anyone with a webcam. Unlike traditional video-based courses, SignSense uses computer vision to analyze users' signs in real-time, providing instant feedback on accuracy—a breakthrough that addresses the critical gap in existing ASL learning tools.

"Learning ASL from videos is like learning to play piano by watching someone else play," said [Team Lead Name], co-founder of SignSense. "You need feedback to improve. SignSense is the first platform that actually watches you sign and tells you if you're doing it right."

**The Problem**

Over 500,000 Americans use ASL as their primary language, and millions more want to learn. But ASL education faces a unique challenge: it's a visual language that requires precise hand shapes, movements, and facial expressions. Traditional apps can teach vocabulary through videos, but they can't verify if learners are signing correctly. Professional ASL classes cost $200-500 and require in-person attendance, putting quality education out of reach for many.

**The Solution**

SignSense uses MediaPipe pose estimation and custom machine learning models to track hand positions and movements through any standard webcam. The platform provides:

- **Real-time sign verification** with accuracy scoring
- **Visual feedback** showing exactly where signs need adjustment  
- **Structured lessons** from fingerspelling to conversational phrases
- **Gamified progress** with streaks, XP, and achievement badges

**User Testimonials**

"I've tried three different ASL apps, and SignSense is the only one that actually made me feel like I was improving," said Maria T., a beta user and mother of a Deaf child. "Being able to see that I'm signing 'thank you' correctly, instead of just hoping I am, is a game-changer."

"As a Deaf educator, I'm thrilled to see technology that takes ASL seriously as a real language," said Dr. James Hernandez, ASL Department Chair at Gallaudet University. "SignSense has the potential to create more allies and more access for our community."

**Availability**

SignSense is available now at signsenese.app with a free tier including alphabet lessons and 20 common signs. Premium subscriptions start at $9.99/month for full curriculum access. Educational institution pricing is available for schools and universities.

**About SignSense**

SignSense was founded by a team of University of Toronto students passionate about using AI for accessibility. The platform launched in [Month 2026] and has already helped over [X] users begin their ASL journey.

**Media Contact:**
[Team Member Name]
press@signsense.app

---

## Appendix: Press Release Iterations

### Iteration 1: Technical Focus (Rejected)
*"SignSense Uses MediaPipe and PyTorch to Enable Real-Time ASL Recognition"*

**Why rejected:** Too technical for general audience. Focused on technology rather than user benefit. Would only appeal to ML engineers, not learners.

### Iteration 2: Social Impact Focus (Refined)
*"New App Bridges Gap Between Deaf and Hearing Communities Through AI-Assisted Learning"*

**Why refined:** Good emotional hook but too broad. Didn't clearly explain the product or differentiation. Added specific problem/solution framing.

### Iteration 3: Duolingo Comparison (Considered)
*"The Duolingo for Sign Language is Finally Here"*

**Why considered but adjusted:** Strong positioning but legally risky and undersells our unique value. We're not just "Duolingo but ASL"—we're solving a fundamentally harder problem with novel technology.

### Final Version: User Benefit + Differentiation
Focused on the core insight that ASL learning lacks feedback, positioned technology as enabler rather than feature, included diverse testimonials to show broad appeal.

---

# ADDITIONAL STRATEGY NOTES

## What Makes This Project Stand Out (Resume Value)

1. **Full-stack complexity:** React + Python + ML + deployment
2. **Real ML work:** Custom model training, not just API calls
3. **Computer vision:** Practical CV application with MediaPipe
4. **Real users:** Can demonstrate traction and user feedback
5. **Social impact:** Accessibility focus is increasingly valued

## Marketing & User Acquisition Strategy

### Phase 1: Validation (Weeks 1-4)
- Post on r/ASL, r/deaf, r/languagelearning for early feedback
- Reach out to ASL instructors for beta testing
- Create TikTok/Instagram content showing the tech in action

### Phase 2: Beta Launch (Weeks 5-8)  
- Product Hunt launch
- University disability services outreach
- ASL student groups at local universities

### Phase 3: Growth (Weeks 9-12)
- SEO content (blog posts: "How to learn ASL online")
- Partnerships with Deaf content creators
- Free tier for educators

## Potential Pivots if Needed

1. **Narrow scope:** Focus only on fingerspelling (26 letters) if word recognition is too hard
2. **Different market:** Pivot to ASL assessment tool for employers (compliance training)
3. **Different tech:** Use server-side inference if browser ML is too slow

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model accuracy too low | Start with fingerspelling (proven feasible), expand vocabulary gradually |
| Users don't come | Partner with ASL educators for captive audience |
| Technical complexity | Use MediaPipe's pre-built hand tracking, focus innovation on feedback UX |
| Scope creep | MVP = alphabet + 20 words + feedback. Everything else is bonus |

---

## Next Steps for Team

1. [ ] Fill in team member names and personalize interest statements
2. [ ] Finalize project name (SignSense, SignFlow, or other)
3. [ ] Set up GitHub repo and project board
4. [ ] Begin MediaPipe prototype this week
5. [ ] Divide responsibilities based on interest statements
