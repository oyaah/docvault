# Information Security Policy

## Access Control

### Authentication
All employees must use multi-factor authentication (MFA) for all company systems. Supported MFA methods:
- Hardware security keys (preferred)
- TOTP authenticator apps
- SMS verification (legacy, being phased out)

Passwords must be at least 16 characters and must not be reused across services.

### Authorization
Access follows the principle of least privilege. Employees receive access only to systems required for their role. Access reviews are conducted quarterly.

### VPN
All remote access to internal systems requires VPN connection. The VPN uses WireGuard protocol. Split tunneling is disabled.

## Data Classification

### Levels
- **Public**: Marketing materials, blog posts, open-source code
- **Internal**: Internal wikis, team documents, meeting notes
- **Confidential**: Customer data, financial records, HR records
- **Restricted**: Encryption keys, security audit reports, incident reports

### Handling
- Confidential and Restricted data must be encrypted at rest and in transit
- Restricted data must not be stored on personal devices
- All data must be classified before sharing externally

## Incident Response

### Reporting
All security incidents must be reported to security@company.com within 1 hour of discovery. Do not attempt to investigate or remediate on your own.

### Severity Levels
- **P1 (Critical)**: Active data breach, ransomware. Response time: 15 minutes.
- **P2 (High)**: Unauthorized access, malware. Response time: 1 hour.
- **P3 (Medium)**: Phishing attempts, policy violations. Response time: 4 hours.
- **P4 (Low)**: Informational, compliance questions. Response time: 24 hours.

### Post-Incident
All P1 and P2 incidents require a post-mortem within 48 hours. Post-mortems are blameless and focus on systemic improvements.

## Device Security

### Company Devices
- Full disk encryption required (FileVault/BitLocker)
- Automatic OS updates enabled
- Endpoint detection and response (EDR) agent installed
- Screen lock after 5 minutes of inactivity

### Personal Devices (BYOD)
Personal devices may not be used to access Confidential or Restricted data. For Internal data access:
- Device must be registered with IT
- MDM profile must be installed
- Latest OS version required

## Compliance

### SOC 2
We maintain SOC 2 Type II compliance. Annual audits are conducted by an independent firm. All employees must complete SOC 2 awareness training annually.

### GDPR
For EU customer data:
- Data processing agreements required with all sub-processors
- Right to deletion requests must be fulfilled within 30 days
- Data breach notification to supervisory authority within 72 hours
