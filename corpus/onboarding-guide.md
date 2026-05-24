# Onboarding Guide

**Policy Owner:** People Operations  
**Last Updated:** 2025-09-01  
**Effective Date:** 2023-01-01  
**Review Cycle:** Semi-annual  
**Applies To:** All new hires (full-time and part-time)  

---

## 1. Welcome to Acme Corp

Congratulations on joining the team. This guide is your roadmap for the first 90 days. It covers everything from setting up your laptop to shipping your first contribution.

Onboarding at an all-remote company is different from walking into an office on your first day. There's no desk with a welcome package, no hallway introductions, no "just follow me around for a week." Instead, we've built a structured, self-directed onboarding program that gives you the context you need without the chaos.

### 1.1 Your Onboarding Team

You'll interact with several people during onboarding:

| Role | Person | Purpose |
|---|---|---|
| **Manager** | Your direct manager | Sets goals, provides context, conducts check-ins |
| **Onboarding Buddy** | A peer on your team (assigned before Day 1) | Informal guide for questions you'd feel weird asking your manager |
| **People Ops Coordinator** | Assigned from People Operations | Handles logistics, benefits enrollment, compliance |
| **IT Provisioner** | From the IT team | Equipment, account setup, security onboarding |

Your onboarding buddy is not a mentor — they're the person you Slack when you need to know which channel to post in, how to submit an expense, or why the deploy pipeline takes 40 minutes. No question is too small.

## 2. Before Day 1 (Pre-boarding)

### 2.1 What We Send You

After you sign your offer letter, the following happens:

| Item | Timeline | Notes |
|---|---|---|
| Welcome email | Within 24 hours of signed offer | Includes start date confirmation and logistics |
| Equipment shipment | 7-10 business days before start date | Laptop, YubiKey, stickers |
| Account provisioning | 1-2 business days before start date | Google Workspace, Slack, Okta invitations arrive via personal email |
| Onboarding schedule | 3 business days before start date | Your Week 1 calendar, pre-populated |
| Buddy introduction | 3 business days before start date | Email intro to your onboarding buddy |

### 2.2 What We Need from You

Before your start date, please complete:

1. **Background check authorization** (via Checkr — link in your welcome email)
2. **Tax forms** (W-4 for US, equivalent for your country)
3. **Direct deposit information** (via BambooHR)
4. **Emergency contact** (via BambooHR)
5. **Equipment delivery confirmation** (reply to the IT email when your laptop arrives)
6. **Profile photo** for Slack and the team directory (headshot, casual is fine)

### 2.3 Prepare Your Workspace

Before Day 1, set up your home office:

- Desk and chair (use your $1,500 home office stipend — see Remote Work Policy)
- Reliable internet (minimum 50 Mbps down / 10 Mbps up)
- A quiet space for video calls
- Your new laptop, charged and powered on

## 3. Week 1: Foundations

### 3.1 Day 1 — Monday

| Time (your local) | Activity | Duration | With |
|---|---|---|---|
| 09:00 | Welcome session | 60 min | People Ops Coordinator |
| 10:30 | Security onboarding | 45 min | IT team |
| 11:30 | Tool setup (self-paced) | 90 min | Solo (IT available in #help-it) |
| 14:00 | Meet your manager | 30 min | Manager |
| 14:30 | Meet your onboarding buddy | 30 min | Buddy |

**Welcome session covers:**
- Company history, mission, and values
- Org structure overview
- Benefits enrollment walkthrough
- Handbook orientation — where to find things
- Q&A

**Security onboarding covers:**
- YubiKey setup and MFA enrollment
- 1Password setup and credential migration
- VPN (Tailscale) installation
- MDM enrollment
- Security policy acknowledgment

**Tool setup checklist:**

- [ ] Google Workspace — confirm email, calendar, drive access
- [ ] Slack — join required channels (see Section 3.3)
- [ ] Okta — verify SSO access to all provisioned apps
- [ ] BambooHR — complete profile, upload photo
- [ ] 1Password — import any necessary shared vault credentials
- [ ] Notion — access team workspace
- [ ] GitLab — confirm account and group membership (Engineering)
- [ ] Figma — confirm access (Design)
- [ ] Expensify — link to Brex card

### 3.2 Day 2 — Tuesday

| Time | Activity | Duration | With |
|---|---|---|---|
| 09:00 | Team introduction meeting | 30 min | Full team |
| 10:00 | Product overview | 60 min | Product Manager |
| 11:30 | Buddy lunch (virtual) | 30 min | Buddy |
| 14:00 | Role-specific orientation (see Section 4) | 90 min | Manager or designee |

### 3.3 Required Slack Channels

All new hires should join:

| Channel | Purpose |
|---|---|
| #announcements | Company-wide announcements (read-only) |
| #general | General discussion |
| #random | Off-topic, social, fun |
| #help-it | IT support |
| #help-finance | Finance and expense questions |
| #help-people-ops | HR and benefits questions |
| #new-hires | Current cohort of new hires — your onboarding community |
| #handbook-updates | Notifications when the handbook changes |
| Your team channel | e.g., #team-platform, #team-growth-marketing |

Optional but recommended:
- #social-pets, #social-cooking, #social-fitness, #social-gaming, #social-parents
- #learning — book clubs, course recommendations, conference recaps
- #kudos — public recognition and appreciation

### 3.4 Day 3 — Wednesday

Focus: Deep dive into your function.

**Engineering:** Codebase walkthrough (architecture, key services, local dev setup, CI/CD pipeline)  
**Design:** Design system review, research repository, design critique process  
**Sales:** CRM walkthrough, sales process, competitive landscape overview  
**Marketing:** Brand guidelines, content calendar, analytics tools  
**People Ops:** HRIS systems, process documentation, current initiatives  

### 3.5 Day 4 — Thursday

| Time | Activity | Duration | With |
|---|---|---|---|
| 09:00 | First 1:1 with your manager | 45 min | Manager |
| 10:00 | Self-paced handbook reading | 120 min | Solo |
| 14:00 | Cross-functional introduction | 30 min | Key stakeholders from adjacent teams |

**First 1:1 agenda:**

1. How are you feeling? What's clear? What's confusing?
2. Review 30/60/90 day goals (manager provides a draft, you collaborate)
3. Working style preferences — when do you work best? How do you prefer feedback?
4. Communication preferences — Slack vs. video for quick questions?
5. Questions about the team, the projects, the company

### 3.6 Day 5 — Friday

No scheduled meetings. This is your day to:

- Read the handbook sections relevant to your role
- Explore the codebase / tools / systems
- Set up your local development environment (Engineering)
- Read recent project documentation and team meeting notes
- Write down questions for next week

## 4. Role-Specific Onboarding

### 4.1 Engineering

**Week 1-2 deliverables:**
- [ ] Local development environment running with all services
- [ ] Successfully run the test suite locally
- [ ] Complete the "Good First Issue" task (pre-selected by your manager)
- [ ] Submit your first merge request
- [ ] Complete the Secure Coding Practices training

**Key documentation to read:**
- Architecture Decision Records (ADRs) — last 10
- Incident retrospectives — last 5
- On-call runbook for your team
- Coding standards and style guide
- CI/CD pipeline documentation

**First month milestone:** Ship a meaningful feature or bug fix to production independently.

### 4.2 Design

**Week 1-2 deliverables:**
- [ ] Audit 3-5 existing user flows and document observations
- [ ] Attend a design critique session (observe only)
- [ ] Set up your Figma workspace with team libraries
- [ ] Complete the accessibility training module

**First month milestone:** Own a design task from brief to developer handoff.

### 4.3 Sales

**Week 1-2 deliverables:**
- [ ] Shadow 5 discovery calls and 3 demos
- [ ] Complete product certification (internal assessment)
- [ ] Build a list of 50 target accounts in your territory
- [ ] Write your first prospecting email sequence (reviewed by manager)

**First month milestone:** Conduct your first solo discovery call.

### 4.4 Customer Success

**Week 1-2 deliverables:**
- [ ] Shadow 5 customer calls
- [ ] Complete product certification
- [ ] Review 10 recent support tickets to understand common issues
- [ ] Meet with your assigned customers' previous CSM for handoff

**First month milestone:** Run your first customer QBR independently.

## 5. 30/60/90 Day Framework

### 5.1 First 30 Days: Learn

**Goal:** Understand the landscape — people, product, processes.

| Outcome | How You Know You're There |
|---|---|
| Understand the product | Can demo the core workflow to a colleague |
| Know the team | Had a 1:1 coffee chat with every team member |
| Understand the processes | Can navigate the handbook without help |
| Deliver something small | Shipped one contribution (role-dependent) |
| Build relationships | Connected with 3+ people outside your team |

**30-day check-in with your manager:**
- What have you learned?
- What surprised you?
- What's still unclear?
- Are the 60/90 day goals still the right ones?

### 5.2 Days 31-60: Contribute

**Goal:** Work independently on defined tasks. Start adding value.

| Outcome | How You Know You're There |
|---|---|
| Independent work | Completing tasks without step-by-step guidance |
| Proactive communication | Providing status updates before being asked |
| Process awareness | Following team processes correctly |
| Relationship building | Collaborating effectively with adjacent teams |
| Feedback integration | Incorporating feedback from your first review |

**60-day check-in:**
- What are you most proud of?
- Where are you struggling?
- What process or tool frustrates you most? (Fresh eyes are valuable — tell us.)
- What support do you need for the next 30 days?

### 5.3 Days 61-90: Own

**Goal:** Operate at full speed. Own outcomes, not just tasks.

| Outcome | How You Know You're There |
|---|---|
| Full speed | Delivering at the expected pace for your level |
| Ownership | Taking responsibility for outcomes, not just execution |
| Initiative | Identifying improvements without being asked |
| Team contribution | Helping others (reviewing code, sharing knowledge, unblocking peers) |
| Cultural fit | Embodying company values in daily work |

**90-day review:**
- Formal probationary check-in with your manager
- Review of 30/60/90 goals — what was achieved, what shifted
- Set OKRs for the next quarter
- Feedback on the onboarding experience (helps us improve for the next cohort)

## 6. Onboarding Cohorts

New hires start on the **first and third Monday of each month**. Starting with a cohort means you have peers going through the same experience. The #new-hires channel connects the current cohort.

**Cohort activities:**
- Week 1: Group welcome session
- Week 2: CEO Q&A (live, 30 minutes, no topic off-limits)
- Week 3: Cross-functional lunch (random pairings with people outside your team)
- Week 4: Onboarding retrospective (what worked, what didn't — feedback goes directly to People Ops)

## 7. Resources

### 7.1 Key Handbook Pages

| Page | Why It Matters |
|---|---|
| Remote Work Policy | Your workspace, equipment, and schedule |
| Communication Guidelines | How we communicate (read this in the first week) |
| PTO Policy | How to take time off |
| Expense Policy | How to submit expenses and use your stipends |
| Security Practices | Keeping company and customer data safe |
| Performance Reviews | How you're evaluated |
| Career Framework | Levels, tracks, and growth expectations |

### 7.2 Key Tools

| Tool | Purpose | Access |
|---|---|---|
| Google Workspace | Email, calendar, docs | SSO via Okta |
| Slack | Communication | SSO via Okta |
| BambooHR | HR self-service, PTO, payslips | SSO via Okta |
| Expensify | Expense reports | SSO via Okta |
| Notion | Documentation, project management | SSO via Okta |
| GitLab | Code, issues, CI/CD (Engineering) | SSO via Okta |
| Figma | Design (Design team) | SSO via Okta |
| Zoom | Video calls | SSO via Okta |
| 1Password | Credential management | Separate login (set up Day 1) |

### 7.3 Getting Help

| Problem | Where to Go |
|---|---|
| Can't log in to something | #help-it on Slack |
| Benefits question | #help-people-ops on Slack |
| Expense question | #help-finance on Slack |
| Feeling overwhelmed | Your buddy, your manager, or Modern Health |
| Something seems broken/wrong | Your manager or #help-it |
| "Where do I find..." | Your buddy first, then the handbook search |
| Security concern | #security-incidents on Slack |

## 8. Onboarding Feedback

We measure onboarding effectiveness through:

- **New hire survey** at 30 and 90 days (anonymous, 5-minute survey)
- **Manager survey** at 90 days (is the new hire ramping as expected?)
- **Buddy survey** at 30 days (was the buddy program useful?)
- **90-day retention rate** (target: >95%, current: 97%)
- **Time to first contribution** (role-dependent, tracked by team)

Recent improvements based on feedback:

- **2025 Q3:** Added pre-boarding equipment tracking so new hires know exactly when their laptop ships
- **2025 Q2:** Extended the no-meetings Friday to Week 1 (previously started Week 2)
- **2025 Q1:** Introduced the CEO Q&A in Week 2 — rated 4.8/5.0 by new hires
- **2024 Q4:** Switched from assigned buddies to opt-in buddy matching based on interests and time zone

---

*The first 90 days set the trajectory. Take it seriously, ask every question you have, and don't try to absorb everything at once. Nobody expects you to know everything in Week 1 — or Week 12. We're here to help you succeed.*
