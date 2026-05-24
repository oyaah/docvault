# Communication Guidelines

**Policy Owner:** Chief of Staff  
**Last Updated:** 2025-07-10  
**Effective Date:** 2021-03-01  
**Review Cycle:** Annual  
**Applies To:** All team members  

---

## 1. Why Communication Matters More for Us

In a colocated office, communication happens naturally — hallway conversations, whiteboard sessions, overhearing relevant discussions. In an all-remote company, none of this exists. Every piece of context must be deliberately written down and shared. Poor communication in a remote company doesn't just slow things down — it creates an information vacuum that fills with assumptions, misalignment, and duplicated work.

This document defines how we communicate at Acme Corp. It is prescriptive by design. Consistency in communication practices is what makes asynchronous collaboration work at scale.

## 2. Communication Principles

### 2.1 Async-First

Synchronous communication (meetings, calls, real-time chat) is the exception, not the default. Before scheduling a meeting or expecting an instant reply, ask yourself:

- Can this be written in a Slack message or document?
- Does this actually require real-time back-and-forth?
- Am I defaulting to a meeting because it feels easier (for me), even though it's more expensive (for everyone)?

If the answer to the first question is yes, write it down instead.

### 2.2 Write Things Down

If it's not written down, it didn't happen. This applies to:

- **Decisions:** Document the decision, the alternatives considered, and the reasoning
- **Processes:** If you do something more than twice, write a process document
- **Context:** Don't assume the reader knows the backstory. Include links to prior discussions
- **Action items:** Every discussion should end with clear next steps, owners, and deadlines

### 2.3 Low-Context Communication

Write as if the reader:

- Joined the company yesterday
- Has never seen this project before
- Is reading your message at 2 AM in a different time zone without the ability to ask follow-up questions

This means:

- **Spell out acronyms** on first use (even common ones within the company)
- **Link to source documents** rather than saying "as discussed" or "per the plan"
- **State your request explicitly.** "Can you review this MR by Thursday?" not "Thoughts?"
- **Front-load the key point.** The first sentence should tell the reader what they need to know or do

### 2.4 Transparency by Default

Information is public within the company unless there is a specific reason to restrict it. Valid reasons for restricted information:

- Legal or regulatory requirements (e.g., material non-public financial information)
- Personal privacy (e.g., compensation, medical, performance issues)
- Security sensitivity (e.g., vulnerability details before a fix is deployed)
- Competitive sensitivity (explicitly designated by the executive team)

Everything else is shared openly. When in doubt, share.

## 3. Communication Channels

### 3.1 Channel Selection Guide

| Scenario | Channel | Response Expectation |
|---|---|---|
| Project discussion, questions, updates | Slack public channel | 4 business hours |
| Quick logistical question | Slack DM | 4 business hours |
| Formal decision or proposal | Google Doc or Notion page | 48 business hours |
| External communication | Email | 24 business hours |
| Complex discussion needing real-time dialogue | Zoom meeting (scheduled) | N/A |
| Urgent incident | Slack #incidents channel + page on-call | 15 minutes |
| Sensitive personal matter | Slack DM to manager or People Ops | 24 business hours |
| Company announcements | Slack #announcements | Read-only |
| Feedback on policies/processes | Merge request to handbook | 1 week |

### 3.2 Slack

Slack is our primary communication tool. It is not a real-time chat platform — it is an asynchronous messaging system.

#### Channel Naming Conventions

| Prefix | Purpose | Example |
|---|---|---|
| #team- | Team-specific channels | #team-platform, #team-growth |
| #project- | Project-specific channels (temporary) | #project-billing-v2, #project-soc2-2026 |
| #dept- | Department-wide channels | #dept-engineering, #dept-marketing |
| #announce- | Announcements (restricted posting) | #announce-company, #announce-engineering |
| #help- | Help and support channels | #help-it, #help-finance, #help-legal |
| #social- | Social and interest-based | #social-pets, #social-cooking, #social-gaming |
| #ext- | Channels with external guests | #ext-vendor-acme, #ext-client-globex |

#### Slack Norms

1. **Use threads.** Every reply to a message should be in a thread, not in the main channel. Top-level messages in channels are for new topics only.
2. **Use reactions for acknowledgment.** A thumbs-up emoji is faster than "Got it, thanks!" and creates less noise.
3. **Don't use @channel or @here** unless it's genuinely urgent and relevant to everyone in the channel. Most of the time, @-mention specific people.
4. **Set your status.** Keep it current — in a meeting, on PTO, heads-down, at lunch. This replaces the visual cues of an office.
5. **Respect Do Not Disturb.** If someone's DND is on, the message can wait unless it's a P1 incident.
6. **Move long discussions to documents.** If a Slack thread exceeds 15 messages, it should become a Google Doc or Notion page with a link posted back to the thread.
7. **No expectation of reading every message.** Channels generate volume. Catch up on what's relevant; don't try to read everything.

#### Slack Etiquette

- **Don't say "Hi" and wait.** Send the full message. ("Hi" followed by a long pause creates unnecessary anxiety.)
- **Be explicit about urgency.** If it can wait, say so. If it can't, say that too.
- **Avoid naked links.** Add a sentence of context about why you're sharing the link.
- **Edit, don't delete.** If you made a typo or said something wrong, edit the message and note the edit rather than deleting and reposting.

### 3.3 Email

Email is for external communication and formal internal communication only. Internal Slack messages should never be duplicated in email.

Email norms:

- **Subject lines are descriptive.** "[Action Required] Q3 Budget Approval by Oct 15" not "Quick question"
- **One topic per email.** Don't bundle unrelated topics into a single message.
- **Use CC deliberately.** If someone is CC'd, they're informed but not expected to act. If they need to act, put them in the To field with a clear request.
- **Reply-all sparingly.** Most replies do not need to go to the entire distribution.

### 3.4 Google Docs / Notion

For documents that require input from multiple people or will serve as a record of a decision:

- **Google Docs** for collaborative drafting, proposals, and RFCs
- **Notion** for permanent documentation, project management, and structured information

Document norms:

- Every document must have a title, author, date, and status (Draft / In Review / Final)
- Use the commenting feature for feedback — don't make inline edits to someone else's document without permission
- Resolve comments when addressed; don't leave stale comments
- Archive completed documents rather than deleting them

### 3.5 Video Calls (Zoom)

Video calls are for when asynchronous communication won't work. Valid use cases:

- 1:1 meetings between managers and direct reports
- Complex discussions where real-time back-and-forth is genuinely needed
- Team social events and bonding
- Customer and candidate calls
- Incident response coordination

#### Meeting Best Practices

1. **Every meeting has an agenda** shared at least 24 hours in advance. No agenda = attendees may decline.
2. **Meetings are 25 or 50 minutes**, not 30 or 60. This gives people a buffer between meetings.
3. **Start on time, end on time.** If you're not done, schedule a follow-up rather than running over.
4. **Default to cameras on** for meetings with fewer than 10 people. Camera-off is always acceptable for large all-hands or if you're having a rough day — no explanation needed.
5. **One person takes notes.** Rotate the responsibility. Notes are posted in the relevant Slack channel within 2 hours.
6. **Record the meeting** if anyone on the invite list is absent. Recordings are posted in Notion.
7. **The organizer is responsible for action items.** At the end of every meeting, state the action items, owners, and deadlines aloud.

#### Meeting-Free Time

- **No-Meeting Fridays:** Company-wide. No internal meetings on Fridays. External meetings are permitted only when the external party's schedule requires it.
- **Focus blocks:** Team members are encouraged to block 2-4 hours of focus time on their calendar daily. Respect these blocks.

## 4. Writing at Acme Corp

### 4.1 Tone and Voice

- **Professional but human.** We're not writing legal briefs. Write like you'd speak to a respected colleague.
- **Direct, not blunt.** Say what you mean without unnecessary preamble, but be kind.
- **Inclusive language.** Avoid jargon that excludes people outside your function. No "obviously" or "as everyone knows" — these phrases make people feel excluded when they don't know.
- **Active voice.** "The team decided" not "It was decided by the team."

### 4.2 Formatting Standards

- **Use headers** to structure long messages (even in Slack)
- **Use bullet points** for lists of 3+ items
- **Bold key terms** or action items so skimmers catch the important parts
- **Use tables** for comparisons or structured data
- **Keep paragraphs short** — 3-4 sentences maximum

### 4.3 Proposals and RFCs

For significant changes (new tools, process changes, architectural decisions), write a structured proposal:

1. **Problem statement:** What problem are we solving? Why does it matter?
2. **Proposal:** What are you proposing?
3. **Alternatives considered:** What else did you look at? Why was it rejected?
4. **Impact:** Who is affected? What changes for them?
5. **Timeline:** When would this take effect?
6. **Open questions:** What is still unresolved?

Share the document in the relevant Slack channel with a clear deadline for feedback (minimum 5 business days for significant changes).

## 5. Feedback Culture

### 5.1 Giving Feedback

- **Be specific.** "Your MR descriptions have been really thorough lately — it makes reviews much faster" is better than "Good job."
- **Focus on behavior, not character.** "The deploy went out without a rollback plan" not "You're careless about deploys."
- **Timely.** Give feedback within 48 hours of the event. Stale feedback loses impact.
- **Private for constructive, public for positive.** Praise in Slack channels, coach in DMs or 1:1s.

### 5.2 Receiving Feedback

- **Assume positive intent.** The person giving feedback is trying to help you grow.
- **Listen first, respond later.** Don't defend immediately. Take 24 hours if you need to.
- **Thank the person.** Giving feedback is hard. Acknowledge the effort.

### 5.3 Disagreement

Disagreement is healthy. Silence is more dangerous than conflict. When you disagree:

1. **Disagree openly.** Don't nod in the meeting and complain in DMs.
2. **Disagree with data.** Opinions backed by evidence carry more weight.
3. **Commit after the decision.** Once a decision is made, support it fully even if you disagreed. If you can't commit, escalate — don't undermine.

This is the **"disagree and commit"** principle. You can always revisit a decision with new evidence, but passive resistance destroys teams.

## 6. Cross-Functional Communication

### 6.1 Working with Other Teams

When you need something from another team:

1. Post in their team channel (not a DM to an individual)
2. Clearly state what you need, why, and by when
3. Tag the team lead if you're unsure who to ask
4. Respect their prioritization — your request is not automatically their top priority

### 6.2 Escalation Path

If cross-team collaboration isn't working:

1. Direct conversation between the team members involved
2. Involve both team leads
3. Escalate to the shared director or VP
4. Raise in the weekly leadership sync (last resort)

Most issues resolve at step 1 or 2. If you find yourself at step 3 frequently, the root cause is probably a process or incentive misalignment — surface it.

## 7. External Communication

### 7.1 Social Media

Team members are free to have personal social media accounts. When discussing Acme Corp:

- Clearly distinguish personal opinions from company positions
- Do not share confidential or restricted information
- Do not disparage customers, competitors, or partners
- When in doubt, run it by the Communications team

### 7.2 Press and Analyst Inquiries

All media inquiries must be directed to the Communications team (comms@acmecorp.com). Do not respond to press inquiries directly, even to say "no comment."

### 7.3 Public Speaking

Team members are encouraged to speak at conferences and events. When representing Acme Corp:

- Submit the talk topic and abstract to your manager for review
- Do not disclose confidential product roadmap information
- Presentation slides should be reviewed by the Communications team if they include the Acme Corp brand

## 8. Communication Anti-Patterns

Behaviors to avoid:

| Anti-Pattern | Why It's Harmful | Alternative |
|---|---|---|
| "Let's take this offline" (and never doing it) | Decisions die in the gap | Schedule the follow-up immediately or resolve it async in a document |
| Slack message at 11 PM expecting a reply | Creates pressure to be always-on | Schedule the message for business hours or explicitly say "no reply needed tonight" |
| Forwarding a long email chain with "see below" | Forces the recipient to decode context | Summarize the key point and what you need from them |
| Using Slack DMs for decisions | Excludes the team from context | Move decisions to public channels |
| "Quick sync?" with no agenda | Wastes both people's time | State the topic. Often it becomes a Slack message instead |
| Passive-aggressive emoji reactions | Erodes trust | Use words. Say what you mean. |

## 9. Tools Quick Reference

| Tool | Use For | Not For |
|---|---|---|
| Slack | Day-to-day communication, quick questions | Long-form proposals, decisions that need a paper trail |
| Google Docs | Proposals, RFCs, collaborative drafts | Permanent documentation |
| Notion | Project management, permanent docs, meeting notes | Real-time collaboration on drafts |
| Zoom | Synchronous discussions, 1:1s, social events | Anything that could be a Slack message |
| Email | External communication, formal notices | Internal team communication |
| Loom | Async demos, walkthroughs, status updates | Conversations requiring back-and-forth |
| GitLab | Code review, technical RFCs, issue tracking | Non-technical discussions |

---

*Good communication is a skill, not a talent. It improves with deliberate practice. Re-read this document quarterly and honestly assess which norms you've been slipping on.*
