# Policy, Safety & Societal Impact

[← Back to Learning Path](../learning-path.md) | [← Prev: Human-AI Interaction](human-ai-interaction.md) | [📖 Glossary](glossary.md)

**Overview**: Technical capabilities alone don't determine AI's impact on society—policy, governance, and ethical considerations are equally critical. This area steps back to examine the bigger picture: how should AI systems be regulated, what are the existential and near-term risks, how do we ensure equitable access and prevent misuse, and what frameworks exist for responsible AI development? Whether you're building AI systems, advising organizations, or simply want to be an informed citizen, understanding these policy and safety considerations is essential as AI becomes increasingly central to society.

---

## Financial Services & Model Risk Management
**Goal**: Understand regulatory frameworks for AI in financial services

**Why this matters**: Financial institutions were among the first to systematically regulate AI/ML models. The frameworks developed here—particularly around model validation, governance, and risk management—have influenced AI regulation across industries.

1. 📄 [OCC 2011-12: Supervisory Guidance on Model Risk Management](https://web.archive.org/web/20240616095814/https://www.occ.gov/news-issuances/bulletins/2011/bulletin-2011-12a.pdf) (Office of the Comptroller of the Currency, 2011)
   - *Why*: **Foundational regulatory guidance** - establishes the three lines of defense for model risk management; defines what constitutes a "model" and requirements for validation
   - *Key concepts*: Model development, implementation, use; ongoing monitoring and validation; model inventory and governance
   - *Relevance*: Still the primary reference for ML model governance in banking; applicable beyond finance
   - *Note*: OCC and Federal Reserve removed these pages from their live sites; archived copy linked. The FDIC's [adoption notice](https://www.fdic.gov/news/inactive-financial-institution-letters/2017/adoption-supervisory-guidance-model-risk-management) remains live and covers the same joint guidance

2. 📄 [SR 11-7: Guidance on Model Risk Management](https://web.archive.org/web/20250103073254/https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm) (Federal Reserve, 2011)
   - *Why*: Federal Reserve's companion guidance to OCC 2011-12; emphasizes board and senior management oversight
   - *Key addition*: Focuses on governance structure and accountability frameworks
   - *Note*: Identical substantive content to OCC 2011-12 but targets bank holding companies; archived copy linked

3. 📄 [Model Risk Management Handbook](https://web.archive.org/web/20241220200407/https://occ.gov/publications-and-resources/publications/comptrollers-handbook/files/model-risk-management/index-model-risk-management.html) (OCC, Updated 2021)
   - *Why*: Comprehensive implementation guide; covers AI/ML-specific considerations including explainability, data quality, and bias
   - *Practical value*: Real-world procedures for model validation, documentation standards, and audit practices
   - *Updates*: 2021 version explicitly addresses machine learning models and alternative data

4. 📄 [Principles for the Sound Management of Operational Risk](https://www.bis.org/publ/bcbs195.pdf) (Basel Committee on Banking Supervision, 2011)
   - *Why*: International framework for operational risk including model risk
   - *Global perspective*: Harmonizes regulatory expectations across jurisdictions

---

## Data Protection & Privacy Law
**Goal**: Navigate privacy regulations affecting AI systems

**Why this matters**: AI systems are data-hungry, but data collection and use are heavily regulated. Understanding GDPR, CCPA, and related frameworks is essential for legal AI deployment, especially for systems processing personal data.

1. 📄 [General Data Protection Regulation (GDPR)](https://gdpr-info.eu/) (EU, 2018)
   - *Why*: **The global gold standard** for data protection; extraterritorial application affects most AI systems serving EU users
   - *Key AI provisions*: 
     - Article 22: Right to explanation for automated decisions
     - Article 35: Data Protection Impact Assessments for high-risk processing
     - Data minimization and purpose limitation principles
   - *Penalties*: Up to €20M or 4% of global revenue
   - *Note*: [Official text](https://eur-lex.europa.eu/eli/reg/2016/679/oj) | [Practical guide](https://gdpr.eu/)

2. 📄 [California Consumer Privacy Act (CCPA) / California Privacy Rights Act (CPRA)](https://oag.ca.gov/privacy/ccpa) (California, 2020/2023)
   - *Why*: Strongest US state privacy law; CPRA adds specific AI/automated decision-making provisions
   - *AI-specific*: Rights regarding automated decision-making; risk assessments for certain AI systems
   - *Influence*: Model for other US state privacy laws

3. 📄 [AI and Data Protection Convention (Modernized Convention 108+)](https://www.coe.int/en/web/data-protection/convention108-and-protocol) (Council of Europe, 2018)
   - *Why*: First binding international treaty on data protection; addresses AI explicitly
   - *Global scope*: Open to non-European countries

4. 📄 [Framework Convention on Artificial Intelligence and Human Rights, Democracy and the Rule of Law (CETS No. 225)](https://www.coe.int/en/web/artificial-intelligence/the-framework-convention-on-artificial-intelligence) (Council of Europe, 2024)
   - *Why*: **First binding international treaty on AI itself** - extends the Convention 108+ model from data protection to the full AI lifecycle; obliges parties to ensure AI activities are consistent with human rights, democracy and the rule of law
   - *Scope*: Open to non-member states; signatories include the UK, France, Norway and the EU
   - *Status*: Opened for signature 5 September 2024. **Not yet in force** - it requires five ratifications, including at least three Council of Europe member states. The EU became the first Party on 15 May 2026 ([Council Decision (EU) 2026/1080](https://eur-lex.europa.eu/eli/dec/2026/1080/oj)); as of October 2026 it is the only one. Check the Treaty Office's signature and ratification chart for the current count
   - *Contrast*: Sets state-level obligations rather than product requirements, so it complements rather than duplicates the EU AI Act

---

## AI-Specific Legislation & Executive Action
**Goal**: Understand emerging AI-specific regulatory frameworks

**Why this matters**: Unlike sector-specific rules, these frameworks regulate AI systems directly based on risk, use case, or capability level. They represent the future of AI governance.

> **Dates move.** Effective dates in this section have shifted repeatedly — the EU deferred its high-risk deadlines, Colorado repealed a law before it took effect, and New York rewrote one after signing. Everything below is stated as of **October 2026**; verify against the primary source before relying on it. For instruments that are still moving, see the [Emerging & Pending Watchlist](#emerging--pending-watchlist).

### European Union

1. 📄 [EU Artificial Intelligence Act (Regulation (EU) 2024/1689)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) (EU, 2024)
   - *Why*: **World's first comprehensive AI law** - risk-based regulatory framework that has become the reference point most other jurisdictions define themselves against
   - *Risk tiers*:
     - Unacceptable risk: Banned (social scoring, real-time biometric surveillance)
     - High risk: Strict requirements (hiring, credit scoring, law enforcement)
     - Limited risk: Transparency obligations (chatbots must disclose they're AI)
     - Minimal risk: No requirements
   - *Key obligations*: Risk assessment, data quality, documentation, human oversight, accuracy requirements
   - *Foundation models*: Additional requirements for General Purpose AI (GPAI) and systemic risk models
   - *Application timeline* (as amended by the Digital Omnibus, as of October 2026):
     - Prohibited practices and AI literacy duties: 2 February 2025
     - GPAI model obligations: 2 August 2025 (models already on the market have until 2 August 2027); Commission fining powers over GPAI providers (Art. 101: up to €15M or 3% of worldwide turnover, whichever is higher) from 2 August 2026
     - Article 50 transparency duties: 2 August 2026; generative systems already on the market before then have until 2 December 2026 to meet the Art. 50(2) marking duty
     - New Article 5 prohibitions on AI generating non-consensual intimate imagery and CSAM: 2 December 2026
     - Annex III standalone high-risk systems: deferred to 2 December 2027
     - Annex I high-risk AI embedded in regulated products: deferred to 2 August 2028
   - *Digital Omnibus on AI*: Commission proposal of 19 November 2025; adopted as Regulation (EU) 2026/1744 (see below), which produced the deferred dates above
   - *Resources*: [Commission overview](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) | [Article explorer](https://artificialintelligenceact.eu/ai-act-explorer/) (unofficial)

2. 📄 [General-Purpose AI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai) (European Commission, 2025)
   - *Why*: The practical bridge between the AI Act's GPAI obligations and what providers actually have to do; signing it earns streamlined compliance and more predictable enforcement
   - *Structure*: Three chapters — Transparency, Copyright, and Safety and Security
   - *Status*: Published 10 July 2025, ahead of GPAI obligations applying on 2 August 2025; voluntary, with signatories published by the Commission (22 listed as of October 2026; xAI signed only the Safety and Security chapter). The Omnibus amended Art. 56(6) so the Commission now assesses and publishes the Code's adequacy

3. 📄 [Regulation (EU) 2026/1744: Digital Omnibus on AI](https://eur-lex.europa.eu/eli/reg/2026/1744/oj) (EU, 2026)
   - *Why*: **The first major amendment to the AI Act** - resets the high-risk compliance timeline and adds the first new prohibitions since 2024; read it alongside the original Act to see which obligations proved hard to meet on schedule
   - *Key changes*: Defers Annex III high-risk duties to 2 December 2027 and Annex I to 2 August 2028; adds Art. 5 prohibitions on AI generating non-consensual intimate imagery and CSAM (from 2 December 2026); grace period to 2 December 2026 for Art. 50(2) marking of legacy generative systems; gives the AI Office exclusive competence over AI systems built on a provider's own GPAI model
   - *Not changed*: The GPAI model obligations in Articles 53 and 55
   - *Status*: Adopted 8 July 2026; published in the Official Journal 24 July 2026; in force 27 July 2026

4. 📄 [Guidelines on Transparency Obligations for Providers and Deployers of AI Systems (Article 50)](https://digital-strategy.ec.europa.eu/en/library/guidelines-transparency-obligations-providers-and-deployers-ai-systems) (European Commission, 2026)
   - *Why*: The main interpretive text for the AI Act duties most systems will hit first - chatbot disclosure, machine-readable marking of synthetic content, and deepfake labelling
   - *Status*: Published 20 July 2026, ahead of Article 50 applying on 2 August 2026; non-binding

5. 📄 [Code of Practice on Marking and Labelling of AI-Generated Content](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content) (European Commission, 2026)
   - *Why*: The voluntary route for showing compliance with Article 50(2), (4) and (5) - the transparency counterpart to the GPAI Code of Practice
   - *Structure*: Section 1 covers providers (marking and detection); Section 2 covers deployers (labelling deepfakes and AI-generated text)
   - *Status*: Final version published 10 June 2026

### United States — Federal

US federal AI policy reversed direction in January 2025. The entries below are ordered chronologically so the shift is legible; two are retained despite being rescinded because the successor orders are written against them.

6. 📄 [National AI Initiative Act](https://www.ai.gov/) (US, 2021)
   - *Why*: Establishes national AI strategy, funding, and coordination
   - *Key bodies*: National AI Initiative Office; National AI Advisory Committee

7. 📄 [Blueprint for an AI Bill of Rights](https://bidenwhitehouse.archives.gov/ostp/ai-bill-of-rights/) (US, 2022)
   - *Why*: Non-binding principles for AI design and deployment in the US
   - *Five principles*: Safe and effective systems; algorithmic discrimination protections; data privacy; notice and explanation; human alternatives
   - *Practical tools*: [Full text and technical companion (PDF)](https://bidenwhitehouse.archives.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf)
   - *Status*: **Withdrawn** - removed from whitehouse.gov in January 2025; archived copy linked. Retained here for historical context, as it shaped several state laws that outlived it

8. 📄 [Executive Order 14110 on Safe, Secure, and Trustworthy AI](https://www.federalregister.gov/documents/2023/11/01/2023-24283/safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence) (US, 2023)
   - *Why*: The most comprehensive US federal AI policy to date; established reporting requirements for large models and set the 10^26 FLOP threshold that state frontier-AI laws later adopted
   - *Key mandates*:
     - Safety testing for models trained with >10^26 FLOPs
     - Red-teaming requirements
     - Watermarking for AI-generated content
     - Standards development via NIST
   - *Status*: **Revoked** 20 January 2025 by EO 14148; superseded by EO 14179. Downstream agency work it triggered (AI use-case inventories, NIST standards activity) largely survived the revocation
   - *Note*: Federal Register copy linked; the original White House page is gone

9. 📄 [Executive Order 14179: Removing Barriers to American Leadership in Artificial Intelligence](https://www.federalregister.gov/documents/2025/01/31/2025-02172/removing-barriers-to-american-leadership-in-artificial-intelligence) (US, 2025)
   - *Why*: **The pivot point in US federal AI policy** - replaces EO 14110's safety-and-equity framing with an innovation-and-competitiveness mandate; directs agencies to review and roll back prior AI policies
   - *Consequences*: Triggered removal of EEOC and DOL AI guidance; mandated an AI action plan within 180 days
   - *Date*: Signed 23 January 2025

10. 📄 [OMB M-25-21: Accelerating Federal Use of AI through Innovation, Governance, and Public Trust](https://www.whitehouse.gov/wp-content/uploads/2025/02/M-25-21-Accelerating-Federal-Use-of-AI-through-Innovation-Governance-and-Public-Trust.pdf) (Office of Management and Budget, 2025)
    - *Why*: The operative rulebook for how federal agencies actually adopt AI - matters more day-to-day than the executive orders it implements
    - *Scope*: Agency AI governance structures, minimum practices for high-impact AI use cases, Chief AI Officer responsibilities

11. 📄 [Winning the AI Race: America's AI Action Plan](https://www.ai.gov/action-plan) (US, 2025)
    - *Why*: The policy blueprint EO 14179 ordered; sets the administration's AI agenda across innovation, infrastructure and international diplomacy
    - *Notable*: Recommends withholding federal AI funding from states with regulation deemed burdensome - the seed of the preemption fight that followed
    - *Date*: July 2025

12. 📄 [Executive Order 14365: Ensuring a National Policy Framework for Artificial Intelligence](https://www.federalregister.gov/documents/2025/12/16/2025-23092/ensuring-a-national-policy-framework-for-artificial-intelligence) (US, 2025)
    - *Why*: **Opens a direct federal-versus-state conflict over who regulates AI** - declares a policy of a "minimally burdensome" national standard and moves to displace inconsistent state law
    - *Key mechanisms*:
      - DOJ AI Litigation Task Force, operational from 10 January 2026, to challenge state AI laws in federal court
      - Conditions certain federal funding on state regulatory posture
    - *Carve-outs*: Expressly does not target state laws on child safety, AI compute and data-center infrastructure, or state procurement and government use of AI
    - *Legal caveat*: Preemption ordinarily flows from congressional statute, not executive order; commentators widely expect the EO to guide federal conduct rather than independently displace state law
    - *In practice (as of October 2026)*: DOJ has not filed a standalone suit against a state AI law. Instead it [moved to intervene](https://www.justice.gov/opa/pr/justice-department-intervenes-xai-lawsuit-challenging-colorados-algorithmic-discrimination) on 24 April 2026 in *X.AI LLC v. Weiser* (D. Colo., No. 1:26-cv-01515), challenging Colorado's AI law on Equal Protection grounds; an agreed minute order of 27 April bars Colorado enforcement until 14 days after a ruling on xAI's preliminary-injunction motion. The FTC's [proposed policy statement on AI accuracy](https://www.ftc.gov/news-events/news/press-releases/2026/07/ftc-seeks-public-comment-policy-statement-addressing-ai-accuracy) (1 July 2026) treats undisclosed steering of AI outputs as potentially deceptive and argues the Colorado AI Act is impliedly preempted. Congress has enacted no preemption statute
    - *Date*: Signed 11 December 2025; published in the Federal Register 16 December 2025

13. 📄 [OMB M-26-04: Increasing Public Trust in Artificial Intelligence Through Unbiased AI Principles](https://www.whitehouse.gov/wp-content/uploads/2025/12/M-26-04-Increasing-Public-Trust-in-Artificial-Intelligence-Through-Unbiased-AI-Principles-1.pdf) (Office of Management and Budget, 2025)
    - *Why*: Turns EO 14319 ("Preventing Woke AI in the Federal Government", July 2025) into binding procurement requirements for agencies buying LLMs - the clearest example of the administration regulating model behaviour through purchasing power rather than statute
    - *Date*: 11 December 2025; M-25-21 and M-25-22 remain in effect alongside it

14. 📄 [Executive Order 14409: Promoting Advanced Artificial Intelligence Innovation and Security](https://www.federalregister.gov/documents/2026/06/05/2026-11415/promoting-advanced-artificial-intelligence-innovation-and-security) (US, 2026)
    - *Why*: **The first federal channel for pre-release government review of frontier models** - routed through national-security agencies and kept voluntary, in contrast to the mandatory reporting of the revoked EO 14110
    - *Key provisions*: Directs agencies to design, within 60 days, a voluntary framework under which developers can give the government access to "covered frontier models" for up to 30 days before release to other trusted partners; the NSA Director designates covered models in consultation with others; expressly rules out mandatory licensing or preclearance
    - *Date*: Signed 2 June 2026; published in the Federal Register 5 June 2026

15. 📄 [Executive Order 14434: Inaugurating the Era of Super Intelligence](https://www.federalregister.gov/documents/2026/10/02/2026-20321/inaugurating-the-era-of-super-intelligence) (US, 2026)
    - *Why*: Largely terminological, but it changes the vocabulary of every future federal AI document and starts a push for a statutory definition - expect "SI" to appear in agency guidance from late 2026
    - *Key provisions*: Agencies must use "Super Intelligence" and "SI" in place of "Artificial Intelligence" and "AI" where the law allows; existing regulations, contracts and historical records are exempt; "SI" takes the meaning of "AI" in 15 U.S.C. 9401(3); OSTP to propose legislative language defining the term within 60 days
    - *Date*: Signed 29 September 2026; published in the Federal Register 2 October 2026

### United States — State Laws

With federal AI-specific legislation stalled, states became the primary source of binding US AI law. The entries below are the first-of-kind statutes that other states are modelling; see the trackers under [Practical Compliance](#practical-compliance--implementation) for the full and fast-moving picture.

16. 📄 [California SB 53: Transparency in Frontier Artificial Intelligence Act (TFAIA)](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202520260SB53) (California, 2025)
    - *Why*: **First US state statute targeting frontier AI safety and transparency** - the template New York and others followed
    - *Key requirements*:
      - Publish a frontier AI framework describing safety practices
      - Publish a transparency report before releasing a new frontier model
      - Transmit catastrophic-risk assessments to the California Office of Emergency Services
      - Report critical safety incidents to Cal OES
      - Whistleblower protections for covered employees
    - *Scope*: "Frontier model" = foundation model trained on >10^26 integer or floating-point operations, including subsequent fine-tuning or material modification
    - *Status*: Signed 29 September 2025; effective 1 January 2026

17. 📄 [California SB 243: Companion Chatbots](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202520260SB243) (California, 2025)
    - *Why*: **First US law imposing safety duties on companion AI**, enacted in response to incidents involving minors and conversational AI; the model for a wave of 2026 chatbot bills
    - *Key requirements*: Disclose that the user is interacting with AI; for known minors, break reminders at least every three hours; maintain a protocol addressing suicidal ideation and self-harm content with referral to crisis services; annual reporting
    - *Status*: Signed 13 October 2025; effective 1 January 2026. Its child-safety provisions were extended by SB 1119 (below)

18. 📄 [California SB 1119: Companion Chatbots — Children's Safety](https://www.gov.ca.gov/2026/09/10/governor-newsom-signs-the-strongest-child-safety-chatbot-and-social-media-laws-in-the-nation/) (California, 2026)
    - *Why*: The second-generation companion-chatbot law - builds on SB 243 by moving from disclosure and break reminders to affirmative child-safety duties, the direction most 2026 chatbot bills are heading
    - *Status*: Signed 10 September 2026; main duties apply from 1 July 2027

19. 📄 [Texas HB 149: Responsible Artificial Intelligence Governance Act (TRAIGA)](https://capitol.texas.gov/BillLookup/History.aspx?LegSess=89R&Bill=HB149) (Texas, 2025)
    - *Why*: A deliberately narrower alternative to the EU-style risk framework - **intent-based** prohibitions rather than impact-based duties, which makes it the reference point for states resisting the Colorado model
    - *Key provisions*: Bans AI intended for behavioral manipulation, unlawful discrimination, incitement to violence, or CSAM generation; largely limits affirmative obligations to government use; first-in-nation AI regulatory sandbox
    - *Enforcement*: Attorney General exclusively, with notice and a 60-day cure period; no private right of action; preempts local AI ordinances
    - *Status*: Signed 22 June 2025; effective 1 January 2026

20. 📄 [Illinois HB 3773: AI in Employment (amending the Illinois Human Rights Act)](https://www.ilga.gov/Legislation/BillStatus?DocNum=3773&GAID=17&DocTypeID=HB&LegID=157807&SessionID=112) (Illinois, 2024)
    - *Why*: Makes discriminatory AI use in employment a civil rights violation under existing state anti-discrimination machinery, rather than building a separate AI regime - a materially different regulatory strategy worth contrasting with Colorado and Texas
    - *Key requirements*: Prohibits AI use that discriminates against protected classes across hiring, promotion, discipline and discharge; requires notice to applicants and employees when AI is used
    - *Status*: Effective 1 January 2026

21. 📄 [Illinois SB 315: Artificial Intelligence Safety Measures Act (Public Act 104-0538)](https://www.ilga.gov/Legislation/BillStatus?DocNum=315&GAID=18&DocTypeID=SB&SessionID=114&GA=104) (Illinois, 2026)
    - *Why*: **First US statute requiring independent third-party audits of frontier AI developers** - goes a step beyond the self-published frameworks and transparency reports of California SB 53 and New York's RAISE Act
    - *Key requirements*: Annual independent third-party audits, with the heaviest duties on large frontier developers
    - *Status*: Signed 6 July 2026; effective 1 January 2027

22. 📄 [Connecticut SB 5: An Act Concerning Online Safety (Public Act 26-15)](https://www.cga.ct.gov/asp/cgabillstatus/cgabillstatus.asp?selBillType=Bill&which_year=2026&bill_num=5) (Connecticut, 2026)
    - *Why*: An omnibus approach - one statute bundling AI companions, automated employment decisions and youth online safety, after the state's earlier comprehensive AI bill (SB 2) repeatedly failed
    - *Status*: Signed 27 May 2026; obligations phase in from 1 October 2026

23. 📄 [Utah SB 149: Artificial Intelligence Policy Act](https://le.utah.gov/~2024/bills/static/SB0149.html) (Utah, 2024)
    - *Why*: **First US state law specifically regulating generative AI** - a minimal disclosure-based approach that predates the comprehensive frameworks
    - *Key requirements*: Disclose generative AI use on request; 2025 amendments (SB 226, SB 332) added proactive disclosure for high-risk interactions involving health, financial or biometric data
    - *Status*: Effective 1 May 2024; amended 2025

24. 📄 [Colorado SB 26-189: Automated Decision-Making Technology](https://leg.colorado.gov/bills/sb26-189) (Colorado, 2026)
    - *Why*: **The cautionary tale of the group** - Colorado passed the first comprehensive US state AI law (SB 24-205, 2024), delayed it twice, then repealed and replaced it before it ever took effect. Reading the two together shows what proved unworkable about impact-assessment-based regulation
    - *What changed*: Replaces the "high-risk AI system" reasonable-care and impact-assessment regime with a narrower transparency and disclosure framework built on "covered automated decision-making technology" that materially influences consequential decisions
    - *Key requirements*: Pre-use disclosure; explanation of adverse outcomes within 30 days; meaningful human review; documentation of training data categories and system constraints
    - *Status*: Signed 14 May 2026; substantive obligations commence 1 January 2027. Attorney General rulemaking is under way ([coag.gov/ai](https://coag.gov/ai/)). Enforcement is stayed by agreed order in *X.AI LLC v. Weiser* pending a preliminary-injunction ruling (see EO 14365 above)

25. 📄 [New York S6953B/A6453B: Responsible AI Safety and Education (RAISE) Act](https://www.nysenate.gov/legislation/bills/2025/S6953) (New York, 2025/2026)
    - *Why*: The second state frontier-AI statute; its 2026 rewrite pulled it substantially toward California's SB 53, making the pair a good study in state-law convergence
    - *Key requirements*: Safety and security protocols for frontier models; incident reporting; annual protocol review; published transparency documentation with trade-secret redactions
    - *Scope*: Developers with annual revenue over $500M, for models trained with >10^26 operations and over $100M in compute cost
    - *Oversight*: Rulemaking assigned to a new office within the New York Department of Financial Services
    - *Status*: Originally signed 19 December 2025; the operative text is the **chapter amendment ([S8828](https://www.nysenate.gov/legislation/bills/2025/S8828), Chapter 96) signed 27 March 2026**. Effective 1 January 2027 - one year after SB 53

### Other Jurisdictions

26. 📄 [China's Generative AI Regulations](http://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm) (China, 2023)
    - *Why*: First national regulation specifically for generative AI
    - *Requirements*: Content filtering, factual accuracy, labeling of AI-generated content
    - *Philosophical approach*: Content-focused vs. Western risk-focused frameworks
    - *English summary*: [Stanford HAI](https://hai.stanford.edu/policy/chinas-generative-ai-regulations)

27. 📄 [Cybersecurity Law of the PRC, 2025 Amendment (new Article 20 on AI)](https://www.gov.cn/yaowen/liebiao/202510/content_7046194.htm) (NPC Standing Committee, 2025)
    - *Why*: Writes AI into national statute for the first time - support for AI research and infrastructure alongside ethics rules and risk monitoring - in place of the stalled single comprehensive AI law
    - *Status*: Adopted 28 October 2025; in force 1 January 2026. The first revision of the 2016 law

28. 📄 [Measures for Labeling AI-Generated Synthetic Content](https://www.cac.gov.cn/2025-03/14/c_1743654684782215.htm) (Cyberspace Administration of China, 2025)
    - *Why*: Turns AI content labeling from principle into a concrete build requirement - **both visible labels and embedded metadata/watermark provenance**, which is further than any Western regime has gone in binding law
    - *Status*: Effective 1 September 2025
    - *Context*: Pairs with algorithm registration and security self-assessment obligations already in force

29. 📄 [Interim Measures for the Administration of Anthropomorphic AI Interactive Services](https://www.cac.gov.cn/2026-04/10/c_1777558395078289.htm) (Cyberspace Administration of China et al., 2026)
    - *Why*: **The first binding national regulation aimed specifically at AI companions and emotionally engaging chatbots** - a useful contrast with the US state chatbot laws that emerged at the same time
    - *Key provisions*: Article 14 bars providing minors with "virtual relative" or "virtual partner" services; guardian consent for under-14s; a mandatory minors mode
    - *Status*: CAC Order No. 21, issued jointly by five agencies on 10 April 2026; effective 15 July 2026

30. 📄 [Framework Act on Artificial Intelligence Development and Trust Base Creation](https://www.korea.kr/news/policyNewsView.do?newsId=148958380) (South Korea, 2025)
    - *Why*: **Second jurisdiction after the EU with a comprehensive horizontal AI statute** - the clearest test of whether the EU model travels
    - *Key requirements*: Obligations for high-impact AI across designated sectors; fairness and non-discrimination duties; AI-generated content labeling
    - *Status*: Effective 22 January 2026; Enforcement Decree is Presidential Decree No. 36053. The government has deferred both fines and fact-finding investigations for at least one year, except in extreme cases such as loss of life or serious human-rights harm
    - *Note*: Link is the Korean government's announcement; the law.go.kr English search previously linked here returns no results

31. 📄 [Act on the Promotion of Research, Development and Utilization of AI-Related Technologies (Act No. 53 of 2025) and AI Basic Plan](https://www8.cao.go.jp/cstp/ai/ai_plan/ai_plan.html) (Japan, 2025)
    - *Why*: Japan's deliberately **promotion-first** statute - no penalties, with governance delivered through a Cabinet-level strategy and plan; the clearest Asian counterpoint to the Korean and EU models
    - *Status*: Promulgated 4 June 2025; first AI Basic Plan adopted by Cabinet 23 December 2025; revised plan adopted 14 July 2026

32. 📄 [Law on Artificial Intelligence No. 134/2025/QH15](https://vanban.chinhphu.vn/?pageid=27160&docid=216334&classid=1&typegroupid=3) (Vietnam, 2025)
    - *Why*: Southeast Asia's first comprehensive, risk-based AI statute - evidence that horizontal AI laws are spreading beyond the EU and Korea
    - *Status*: Adopted 10 December 2025; in force 1 March 2026

33. 📄 [India AI Governance Guidelines](https://www.indiaai.gov.in/) (Ministry of Electronics and IT, 2025)
    - *Why*: The most prominent **light-touch** alternative - explicitly declines a standalone AI act, relying on existing law plus voluntary measures; useful contrast to the EU and Korean approaches
    - *Status*: Released November 2025. Not enforceable law; AI is governed through the Digital Personal Data Protection Act and sector rules
    - *Date*: Released 5 November 2025. [Official PDF](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc2025115685601.pdf) (hosted by the Press Information Bureau; the IndiaAI portal reorganizes frequently)
    - *Binding follow-up*: The IT (Intermediary Guidelines and Digital Media Ethics Code) Amendment Rules 2026, G.S.R. 120(E) - notified 10 February 2026, in force 20 February 2026 - require labels and provenance metadata on "synthetically generated information" and cut takedown windows to 3 hours (2 hours for non-consensual intimate imagery). The MeitY and e-Gazette pages blocked automated checks, so these details were confirmed through search results

34. 📄 [Brazil PL 2338/2023: Proposed AI Framework](https://www25.senado.leg.br/web/atividade/materias/-/materia/157233) (Brazil, introduced 2023; pending)
    - *Why*: Closely tracks the EU AI Act's risk-based structure; the leading test of whether that model is adopted across Latin America
    - *Status*: **Not yet law.** Approved by the Senate 10 December 2024. As of October 2026 it is awaiting the rapporteur's opinion in a Chamber of Deputies special committee ([case record](https://www.camara.leg.br/proposicoesWeb/fichadetramitacao?idProposicao=2487262)); the rapporteur said on 24 August 2026 that a vote would come after the October elections. Included here as pending legislation, not a binding instrument

35. 📄 [Law No. 132 of 23 September 2025: Provisions and Delegations on Artificial Intelligence](https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:legge:2025-09-23;132) (Italy, 2025)
    - *Why*: **The first EU member state to pass its own national AI law** alongside the AI Act - shows what national legislators add on top of the EU regulation (sector rules, designated authorities, criminal provisions)
    - *Status*: Published in Gazzetta Ufficiale no. 223 on 25 September 2025; in force 10 October 2025

36. 📄 [Law of the Republic of Kazakhstan on Artificial Intelligence (No. 230-VIII)](https://www.akorda.kz/ru/glavoy-gosudarstva-podpisan-zakon-respubliki-kazahstan-ob-iskusstvennom-intellekte-17105536) (Kazakhstan, 2025)
    - *Why*: The first comprehensive AI law in Central Asia, with a risk-tiered structure - another data point on how far the EU-style template is travelling
    - *Status*: Signed 17 November 2025; in force 18 January 2026 (60 days after official publication)

---

## Risk Management Frameworks & Standards
**Goal**: Learn structured approaches to AI risk management

**Why this matters**: Regulations tell you what to do; frameworks tell you how. These voluntary standards provide practical implementation guidance and are increasingly referenced by regulators.

1. 📄 [NIST AI Risk Management Framework (AI RMF)](https://www.nist.gov/itl/ai-risk-management-framework) (NIST, 2023)
   - *Why*: **The definitive US AI risk management framework** - voluntary but increasingly expected by regulators
   - *Structure*: Four core functions: Govern, Map, Measure, Manage
   - *Risk types*: Covers technical, societal, legal, reputational risks
   - *Practical tools*: [Playbook](https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook)
   - *Companion docs*: Addresses trustworthiness (fairness, robustness, transparency, etc.)

2. 📄 [NIST AI 600-1: Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) (NIST, 2024)
   - *Why*: **The companion profile that makes the AI RMF usable for generative systems** - the framework alone predates the GenAI risk surface
   - *Risks addressed*: Confabulation, harmful content generation, privacy, cybersecurity, intellectual property
   - *Structure*: Suggested actions organized around governance, content provenance, pre-deployment testing and incident disclosure, mapped back to the RMF's GOVERN/MAP/MEASURE/MANAGE functions
   - *Date*: Published 26 July 2024

3. 📄 [Center for AI Standards and Innovation (CAISI)](https://www.nist.gov/aisi) (NIST, 2025)
   - *Why*: The US government's technical AI evaluation body; its outputs feed the standards regulators reference
   - *Note*: Renamed from the US AI Safety Institute in mid-2025 - the rename tracked the broader federal shift from safety framing to standards-and-innovation framing
   - *Recent work*: AI Agent Standards Initiative (February 2026); pre-deployment testing agreements with Google DeepMind, Microsoft and xAI (May 2026), adding to earlier OpenAI and Anthropic agreements; published evaluations of foreign frontier models such as [DeepSeek V4 Pro](https://www.nist.gov/news-events/news/2026/05/caisi-evaluation-deepseek-v4-pro) (May 2026); draft NIST IR 8596 Cyber AI Profile (initial preliminary draft, December 2025)

4. 📄 [ISO/IEC 42001: AI Management System](https://www.iso.org/standard/81230.html) (ISO/IEC, 2023)
   - *Why*: International certifiable standard for AI management systems
   - *Scope*: Organizational governance, not individual models
   - *Certification*: Organizations can be ISO 42001 certified
   - *Note*: [Purchase required](https://www.iso.org/standard/81230.html) | [Overview available](https://www.iso.org/artificial-intelligence/ai-management-systems)

5. 📄 [ISO/IEC 42005: AI System Impact Assessment](https://www.iso.org/standard/44545.html) (ISO/IEC, 2025)
   - *Why*: Gives the impact-assessment step a standardized shape - increasingly the mechanism regulations point to when they require one
   - *Relationship*: Designed to be used alongside ISO/IEC 42001
   - *Note*: Purchase required

6. 📄 [ISO/IEC 42006: Requirements for Bodies Providing Audit and Certification of AI Management Systems](https://www.iso.org/standard/42006) (ISO/IEC, 2025)
   - *Why*: Sets the rules for the auditors themselves - without it, an ISO/IEC 42001 certificate means little; it underpins the emerging market for AI management-system certification
   - *Date*: Published July 2025; purchase required

7. 📄 [ISO/IEC 23053: Framework for AI Systems Using ML](https://www.iso.org/standard/74438.html) (ISO/IEC, 2022)
   - *Why*: Technical standard for ML system development lifecycle
   - *Coverage*: Data quality, model development, deployment, monitoring
   - *Note*: [Purchase required](https://www.iso.org/standard/74438.html)

8. 📄 [IEEE 7000 Series on AI Ethics](https://standards.ieee.org/initiatives/autonomous-intelligence-systems/standards/) (IEEE, 2021, ongoing)
   - *Why*: Technical standards for embedding ethics into system design
   - *Key standards*:
     - IEEE 7000: Systems engineering for ethical concerns
     - IEEE 7001: Transparency of autonomous systems
     - IEEE 7002: Data privacy
   - *Approach*: Values-based engineering

9. 📄 [OECD AI Principles](https://oecd.ai/en/ai-principles) (OECD, 2019)
   - *Why*: First intergovernmental AI policy standards; adopted by 40+ countries
   - *Five principles*: Inclusive growth; human-centered values; transparency; robustness; accountability
   - *Implementation*: [National AI policies dashboard](https://oecd.ai/en/dashboards)

10. 📄 [Hiroshima AI Process Reporting Framework 2.0](https://oecd.ai/en/haip-2-launch) (OECD / G7, 2026)
    - *Why*: The G7's voluntary reporting mechanism for advanced-AI developers, revised to be usable by smaller companies - the main international venue where firms publicly report their risk-management practices
    - *Date*: Launched 28 May 2026

11. 📄 [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026) (Bengio et al., 2026)
    - *Why*: **The shared scientific baseline governments cite** on general-purpose AI capabilities and risks - written by 100+ experts and backed by 30+ countries; it deliberately makes no policy recommendations
    - *Date*: Second edition, published 3 February 2026

12. 📄 [Preliminary Report of the Independent International Scientific Panel on AI](https://www.un.org/independent-international-scientific-panel-ai/en/preliminary-report) (United Nations, 2026)
    - *Why*: The UN's first scientific assessment of AI opportunities, risks and impacts, from the panel set up by General Assembly resolution 79/325 - the evidence base for the UN Global Dialogue on AI Governance
    - *Context*: Co-chaired by Yoshua Bengio and Maria Ressa; presented ahead of the first Global Dialogue session (Geneva, 6-7 July 2026). The full assessment is due in 2027
    - *Date*: Published 1 July 2026

13. 📄 [OWASP Top 10 for Agentic Applications for 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) (OWASP GenAI Security Project, 2025)
    - *Why*: The de facto security risk taxonomy for AI agents (ASI01-ASI10), increasingly mapped onto governance frameworks and procurement checklists
    - *Date*: Released 9 December 2025

---

## Sector-Specific AI Guidance
**Goal**: Navigate industry-specific AI regulations

**Why this matters**: Healthcare AI faces different requirements than financial AI. Understanding sector-specific rules is essential for practical deployment.

### Healthcare & Life Sciences

1. 📄 [FDA's Artificial Intelligence/Machine Learning (AI/ML) Software as a Medical Device Action Plan](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices) (FDA, 2021)
   - *Why*: Regulatory pathway for AI medical devices in the US
   - *Key innovation*: "Predetermined Change Control Plans" for continuously learning models
   - *Requirements*: Clinical validation, performance monitoring, algorithm change protocols
   - *Status note*: FDA's January 2025 draft guidance on AI-enabled device software functions (lifecycle management and marketing submissions) was still a draft as of October 2026

2. 📄 [Considerations for the Regulation of Generative AI-Enabled Medical Devices: Discussion Paper](https://www.fda.gov/news-events/press-announcements/fda-seeks-public-feedback-inform-regulatory-approach-generative-ai-enabled-medical-devices) (FDA, 2026)
   - *Why*: FDA's first framework-level thinking on generative and agentic AI inside medical devices, where the change-control assumptions of the SaMD action plan fit poorly
   - *Status*: Discussion paper (not guidance), released 18 August 2026; comments due 19 October 2026 (docket FDA-2026-N-7874)

3. 📄 [EU Medical Device Regulation (MDR) & In-Vitro Diagnostic Regulation (IVDR)](https://health.ec.europa.eu/medical-devices-new-regulations_en) (EU, 2017/2022)
   - *Why*: AI diagnostic tools must comply; high scrutiny for "black box" algorithms
   - *Risk classification*: Most AI diagnostics are Class IIa or higher

4. 📄 [HIPAA Privacy Rule and AI](https://www.hhs.gov/hipaa/index.html) (US, 1996, ongoing interpretation)
   - *Why*: Governs use of protected health information in AI training and deployment
   - *Key concerns*: De-identification requirements, minimum necessary principle

### Employment & HR

5. 📄 [EEOC Guidance on AI and Title VII Adverse Impact](https://web.archive.org/web/20240527151347/https://www.eeoc.gov/laws/guidance/select-issues-assessing-adverse-impact-software-algorithms-and-artificial) (EEOC, 2023)
   - *Why*: Clarifies how US employment discrimination laws apply to AI hiring tools
   - *Key point*: Employers liable for discriminatory AI, even from third-party vendors
   - *Requirements*: Validation studies, adverse impact analysis
   - *Companion*: [ADA and AI guidance](https://web.archive.org/web/20240529055014/https://www.eeoc.gov/laws/guidance/americans-disabilities-act-and-use-software-algorithms-and-artificial-intelligence) (EEOC, 2022)
   - *Status*: **Withdrawn** - both documents were removed from the EEOC site in 2025 following EO 14179; archived copies linked. The underlying law did not change: Title VII, the ADEA and the ADA still apply to AI-assisted employment decisions. Several states legislated into the resulting federal gap (see Illinois HB 3773 above)

6. 📄 [NYC Local Law 144 (Automated Employment Decision Tools)](https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page) (NYC, 2023)
   - *Why*: First US law specifically regulating AI in hiring
   - *Requirements*: Annual bias audits, notice to candidates, alternative evaluation process

### Criminal Justice

7. 📄 [Algorithmic Accountability in Criminal Justice (Various state laws)](https://www.ncsl.org/technology-and-communication/artificial-intelligence-2025-legislation) (NCSL tracker, 2025)
   - *Why*: Many states restrict or require transparency for risk assessment tools
   - *Examples*: California AB 2542, Wisconsin's Loomis decision

---

## Dual-Use AI & National Security
**Goal**: Understand national security considerations for AI

**Why this matters**: Advanced AI models have both beneficial and harmful uses. Export controls, dual-use regulations, and open vs. closed debates shape what can be built and shared.

1. 📄 [Dual-User Foundation Models with Widely Available Model Weights](https://www.ntia.gov/sites/default/files/publications/ntia-ai-open-model-report.pdf) (NTIA, 2024)
   - *Why*: Government perspective on open model policies and dual-use concerns
   - *Note*: Official NTIA government report - freely available

2. 📄 [Export Controls on AI & Emerging Technologies](https://www.bis.doc.gov/index.php/policy-guidance/advanced-computing-and-semiconductor-manufacturing-items) (US Bureau of Industry and Security, 2022, ongoing)
   - *Why*: Controls on exporting AI chips (GPUs), training techniques, and potentially models
   - *2022 updates*: Restrictions on advanced chip exports to China
   - *2025-2026*: The Biden-era AI Diffusion Rule was announced as rescinded on 13 May 2025 (formal removal from the regulations was not confirmed at time of writing); policy has since moved to targeted rules, such as the [15 January 2026 final rule](https://www.federalregister.gov/documents/2026/01/15/2026-00789/revision-to-license-review-policy-for-advanced-computing-commodities) moving H200-class and lower chips for China and Macau to case-by-case licence review

3. 📄 [NSCAI Final Report](https://web.archive.org/web/20240104154550/https://www.nscai.gov/2021-final-report/) (National Security Commission on AI, 2021)
   - *Why*: Comprehensive US national security strategy for AI
   - *Key recommendations*: Defend democratic values; invest in R&D; build international partnerships
   - *Length*: 750+ pages
   - *Note*: The Commission's domain was decommissioned after it sunset; archived copy linked

4. 📄 [Blueprint for an AI Bill of Rights Concerning National Security Systems](https://www.dni.gov/index.php/newsroom/reports-publications) (US, 2023)
   - *Why*: Principles for AI use in intelligence and defense

---

## Responsible AI & Industry Best Practices
**Goal**: Understand voluntary frameworks and industry initiatives

**Why this matters**: Many organizations operate ahead of regulation. These frameworks represent current best practices and often foreshadow future requirements.

1. 📄 [Partnership on AI Guidelines](https://partnershiponai.org/) (Partnership on AI, 2016, ongoing)
   - *Why*: Multi-stakeholder organization developing AI best practices
   - *Members*: Google, Meta, Microsoft, Amazon, civil society groups
   - *Key work*: AI incident database, responsible practices library

2. 📄 [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (Mitchell et al., 2019)
   - *Why*: Transparency framework for documenting model characteristics
   - *Adoption*: Now widely used; required by some regulations
   - *Template*: [GitHub](https://github.com/huggingface/hub-docs/blob/main/modelcard.md)

3. 📄 [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru et al., 2018)
   - *Why*: Documentation framework for training datasets
   - *Purpose*: Increase transparency about data provenance, composition, biases
   - *Impact*: Influenced EU AI Act data documentation requirements

4. 📄 [AI Incident Database](https://incidentdatabase.ai/) (Partnership on AI, 2020, ongoing)
   - *Why*: Systematic collection of AI system failures and harms
   - *Learning*: Pattern identification across incidents
   - *Examples*: Hiring discrimination, safety failures, privacy breaches

5. 📄 [Microsoft Responsible AI Standard](https://www.microsoft.com/en-us/ai/responsible-ai) (Microsoft, 2022)
   - *Why*: Corporate AI governance framework from major AI provider
   - *Public version*: [Principles and approach](https://www.microsoft.com/en-us/ai/principles-and-approach)
   - *Structure*: Goals, requirements, tools, governance

6. 📄 [Google's AI Principles](https://ai.google/responsibility/principles/) (Google, 2018)
   - *Why*: Early corporate AI ethics framework
   - *Key commitments*: Social benefit, fairness, safety, accountability
   - *Controversies*: Application debates (Project Maven)

7. 📄 [Anthropic Responsible Scaling Policy, Version 3](https://www.anthropic.com/rsp-updates) (Anthropic, 2026)
   - *Why*: The most frequently revised frontier-lab safety policy, and a primary source on how voluntary commitments are evolving - v3 adds Frontier Safety Roadmaps, published Risk Reports and external review
   - *Versions*: v3.0 effective 24 February 2026; v3.4 (8 July 2026) is current as of October 2026

8. 📄 [Google DeepMind Frontier Safety Framework, Version 3.1](https://deepmind.google/frontier-safety/) (Google DeepMind, 2026)
   - *Why*: Read alongside the RSP to compare how labs define capability thresholds; v3.1 adds **Tracked Capability Levels** as an early-warning tier below Critical Capability Levels
   - *Date*: 17 April 2026

---

## Governance Research
**Goal**: Follow the research literature that policy is being built on

**Why this matters**: Many of the mechanisms now appearing in law - third-party audits, incident reporting, compute verification, agent identity - were first worked out in research papers. These recent papers are where the next round of obligations is being designed.

1. [Frontier AI Auditing: Toward Rigorous Third-Party Assessment of Safety and Security Practices at Leading AI Companies](https://arxiv.org/abs/2601.11699) (Brundage et al., 2026)
   - *Why*: Proposes four AI Assurance Levels (AAL-1 to AAL-4) for external audits of frontier labs - the conceptual backdrop to audit mandates such as Illinois SB 315

2. [The 2026 Singapore Consensus on Global AI Safety Research Priorities](https://arxiv.org/abs/2608.14611) (Casper et al., 2026)
   - *Why*: A 120-author consensus on which safety research problems matter most, adding societal resilience and agent risks - a map from technical research to policy needs

3. [How to Catch a GPU: A Taxonomy of Verification and Enforcement Mechanisms for International AI Agreements](https://arxiv.org/abs/2607.22619) (Koopmanschap & Barten, 2026)
   - *Why*: Breaks verification of international compute agreements into three problems - preventing uncontrolled acquisition, detecting compute outside the regime, and preventing escape - and shows the first two are underexplored

4. [Open Problems in AI Incident Governance](https://arxiv.org/abs/2607.05163) (Sidhu et al., 2026)
   - *Why*: Documents how inconsistently existing frameworks define, classify and report AI incidents - essential background for the incident-reporting duties in SB 53, the RAISE Act and the EU AI Act

5. [The Law of Stop: Interruptibility, Injunctions, and the Governance of Agentic AI](https://arxiv.org/abs/2609.22882) (Perez, 2026)
   - *Why*: Treats the ability to halt an AI system as a legal and institutional capacity rather than just a technical feature, and proposes reforms for agentic systems

6. [AI Identity: Standards, Gaps, and Research Directions for AI Agents](https://arxiv.org/abs/2604.23280) (Otsuka et al., 2026)
   - *Why*: Surveys standards and regulation on agent identity and identifies structural gaps, including the absence of any liability framework for autonomous agents

---

## Practical Compliance & Implementation
**Goal**: Apply policy frameworks in real-world scenarios

**Resources for Practitioners**:

### Assessment Tools
- **[NIST AI RMF Playbook](https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook)** - Step-by-step implementation
- **[EU AI Act Compliance Checker](https://artificialintelligenceact.eu/assessment/)** - Determine risk classification
- **[Microsoft HAX Toolkit](https://www.microsoft.com/en-us/haxtoolkit/)** - Human-AI experience design patterns

### Audit & Testing
- **[NIST SP 1270: Managing Bias in AI](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1270.pdf)** - Standard for identifying and managing algorithmic bias
- **[Aequitas](http://aequitas.dssg.io/)** - Open-source bias audit toolkit
- **[AI Verify](https://aiverifyfoundation.sg/)** - Singapore's AI testing framework

### Documentation Templates
- **Model Cards** - [Hugging Face template](https://huggingface.co/docs/hub/model-cards)
- **Data Cards** - [Google template](https://sites.research.google/datacardsplaybook/)
- **AI Impact Assessments** - [Ada Lovelace Institute](https://www.adalovelaceinstitute.org/report/algorithmic-impact-assessment-case-study-healthcare/)

### Legal Resources
- **[Stanford HAI Policy Hub](https://hai.stanford.edu/policy)** - Tracking global AI policy
- **[OECD AI Policy Observatory](https://oecd.ai/)** - International policy database
- **[AlgorithmWatch](https://algorithmwatch.org/)** - Automated decision-making accountability
- **[IAPP Global AI Law and Policy Tracker](https://iapp.org/resources/article/global-ai-legislation-tracker/)** - Jurisdiction-by-jurisdiction summary, updated regularly
- **[OECD AI Incidents and Hazards Monitor](https://oecd.ai/en/incidents)** - Near-real-time incident tracking from news sources

### US State Legislation Trackers
State AI law moves faster than any curated list can. Use these for current status rather than relying on the entries above:
- **[MultiState AI Legislation Tracker](https://www.multistate.ai/artificial-intelligence-ai-legislation)** - All 50 states, updated continuously
- **[NCSL AI Legislation Summary](https://www.ncsl.org/technology-and-communication/artificial-intelligence-2025-legislation)** - Enacted legislation by state and session

**2026 trend lines worth watching**: companion-chatbot safety laws spreading across states (and to China); third-party audits of frontier developers (Illinois SB 315); algorithmic pricing restrictions (California AB 325, New York S.7882, Connecticut HB 8002); second-generation employment-AI rules moving from disclosure to audit and anti-discrimination duties; limits on AI performing licensed clinical functions; and non-consensual deepfake rules, now backed federally by the TAKE IT DOWN Act ([FTC enforcement from 19 May 2026](https://www.ftc.gov/news-events/news/press-releases/2026/05/ftc-begins-enforcing-take-it-down-act)).

### Emerging & Pending Watchlist
Instruments that are signed but not yet operative, still in draft, or too new for a full entry. Status as of **October 2026**; items marked *reported* could not be confirmed against a primary source by two independent checks.

**International**
- **EU** - Draft [Commission guidelines on high-risk classification (Art. 6)](https://digital-strategy.ec.europa.eu/en/policies/guidelines-ai-high-risk-systems): draft published 19 May 2026; final adoption expected by end of 2026
- **Council of Europe** - Framework Convention on AI (CETS 225): needs five ratifications to enter into force; the EU is the only Party so far
- **Brazil** - PL 2338/2023: pending in the Chamber of Deputies; vote signalled for after the October 2026 elections
- **India** - New Delhi Declaration on AI Impact, adopted at the AI Impact Summit (February 2026): non-binding; 88 endorsers at adoption, later 91 (*reported*)
- **UK** - No standalone AI bill in the [13 May 2026 King's Speech](https://www.gov.uk/government/speeches/the-kings-speech-2026); AI measures are spread across other bills, such as a Regulating for Growth Bill with sandbox powers

**US federal**
- **Great American AI Act** - [discussion draft](https://obernolte.house.gov/media/press-releases/obernolte-trahan-release-discussion-draft-great-american-ai-act) (4 June 2026) with three-year preemption of state model-development rules; not formally introduced
- **FRONTIER Act** - [introduced](https://obernolte.house.gov/media/press-releases/obernolte-trahan-introduce-bipartisan-frontier-act-strengthen-oversight) 23 July 2026 (H.R. 9925, *reported*); no committee action found

**US states**
- **California, 2026 session** - Also [signed in September 2026](https://www.gov.ca.gov/2026/09/30/californias-nation-leading-ai-framework-just-got-stronger-governor-newsom-signs-more-first-in-the-nation-worker-protections-and-more/): SB 867 (companion chatbots in toys), SB 813 (independent AI verification organizations), SB 1050 (AI-generated performers in ads), SB 947 (No Robo Bosses Act, operative 1 July 2027), AB 1883 (workplace surveillance tools), AB 1979 and SB 503 (AI in health care), SB 1000 and AB 2713 (AI Transparency Act amendments). Vetoed: SB 903 (AI and mental-health professionals) and AB 2575 (AI in health care services)
- **California CPPA** - [Automated decision-making technology regulations](https://cppa.ca.gov/regulations/ccpa_updates.html): businesses already using ADMT for significant decisions must comply by 1 January 2027
- **Colorado** - [HB 26-1263](https://leg.colorado.gov/bills/hb26-1263), conversational AI service operator requirements: signed 29 May 2026; operator duties from 1 January 2027
- **Washington** - [HB 2225](https://app.leg.wa.gov/billsummary?BillNumber=2225&Year=2026), AI companion chatbots: signed 24 March 2026; effective 1 January 2027
- **Oregon** - [SB 1546](https://olis.oregonlegislature.gov/liz/2026R1/Measures/Overview/SB1546), AI companions: enacted (Chapter 85, 2026)
- **Idaho** - [S1297](https://legislature.idaho.gov/sessioninfo/2026/legislation/S1297/), Conversational AI Safety Act: signed 31 March 2026; effective 1 July 2027

---

## Key Takeaways

- **Multi-jurisdictional complexity**: AI systems often face overlapping regulations across geographies and sectors
- **Risk-based approach emerging**: Most frameworks categorize AI by risk level with proportional requirements — though Texas and India show that a light-touch or intent-based alternative is actively contested, not settled
- **The ground shifts under you**: between 2024 and 2026 the EU deferred its high-risk deadlines (now law via Regulation (EU) 2026/1744), the US federal posture reversed and then moved to preempt the states, Colorado repealed its own landmark law before it took effect, and two federal guidance documents were withdrawn outright. Treat every effective date as provisional and check the primary source
- **Horizontal AI laws are spreading**: after the EU and South Korea, Vietnam, Kazakhstan and Italy passed comprehensive AI statutes in 2025, while Japan and India chose promotion-first or light-touch models
- **US regulation is currently state regulation**: with federal AI legislation stalled, binding US obligations come mostly from states — and the federal preemption effort (EO 14365) has now reached the courts in *X.AI v. Weiser*
- **Companion AI is the fastest-moving target**: California, Washington, Oregon, Idaho, Colorado, Connecticut and China all adopted chatbot or AI-companion rules within about a year
- **Documentation is critical**: Model cards, datasheets, impact assessments increasingly expected or required
- **Human oversight emphasized**: Most frameworks require meaningful human involvement in high-stakes decisions
- **Accountability clearly assigned**: Organizations can't hide behind "the algorithm did it"
- **Transparency vs. IP tension**: Balancing explainability requirements with proprietary interests
- **Continuous monitoring**: One-time validation insufficient; ongoing performance monitoring required
- **Interdisciplinary teams**: Compliance requires legal, technical, and domain expertise

---

## Further Reading

### Books & Reports
- **[Weapons of Math Destruction](https://weaponsofmathdestructionbook.com/)** (Cathy O'Neil, 2016) - Impact of algorithms on society
- **[Atlas of AI](https://www.katecrawford.net/)** (Kate Crawford, 2021) - Material and political dimensions of AI
- **[The Alignment Problem](https://brianchristian.org/the-alignment-problem/)** (Brian Christian, 2020) - AI safety and values

### Policy Tracking
- **[Future of Life Institute AI Policy](https://futureoflife.org/ai-policy/)** - Policy advocacy and tracking
- **[Center for AI and Digital Policy](https://www.caidp.org/)** - Global AI policy monitoring
- **[AI Now Institute](https://ainowinstitute.org/)** - Social implications research

### Academic Centers
- **[Stanford Institute for Human-Centered AI (HAI)](https://hai.stanford.edu/)**
- **[MIT Media Lab Ethics & Governance of AI](https://www.media.mit.edu/groups/ethics-and-governance-of-ai/overview/)**
- **[Oxford Internet Institute AI Ethics](https://www.oii.ox.ac.uk/research/ai-ethics/)**

---

**Related**: [Hardware](hardware.md) | [Safety](safety.md) | [Human-AI Interaction](human-ai-interaction.md)
