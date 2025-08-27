# Snowflake Cortex AI Setup Guide

## ❄️ What is Snowflake Cortex AI?

Snowflake Cortex AI is Snowflake's native AI/ML service that provides:
- **Semantic Search**: Natural language understanding for data queries
- **Text Completion**: AI-powered text generation
- **Embeddings**: Vector representations for similarity search
- **Classification**: Automated data categorization

## 🚀 Benefits of Snowflake Cortex AI

| Feature | Benefit |
|---------|---------|
| **Native Integration** | No external APIs or data movement |
| **Cost Effective** | Included in Snowflake compute costs |
| **Secure** | Data never leaves Snowflake |
| **Scalable** | Handles large datasets efficiently |
| **Real-time** | Instant AI-powered search results |

## 📋 Prerequisites

### 1. Snowflake Account Requirements
- **Enterprise Edition** or higher
- **Cortex AI enabled** on your account
- **Appropriate region** (Cortex AI availability varies by region)

### 2. Required Privileges
```sql
-- Your role needs these privileges:
GRANT USAGE ON WAREHOUSE your_warehouse TO your_role;
GRANT USAGE ON DATABASE your_database TO your_role;
GRANT USAGE ON SCHEMA your_schema TO your_role;
GRANT SELECT ON TABLE ASN TO your_role;
```

### 3. Check Cortex Availability
```sql
-- Test if Cortex is available
SELECT CORTEX_SEARCH('test query', columns => ARRAY_CONSTRUCT('GENDER')) 
FROM ASN LIMIT 1;
```

## 🔧 Setup Steps

### Step 1: Contact Snowflake Support
If Cortex AI is not enabled on your account:

1. **Contact Snowflake Support**:
   - Go to Snowflake Support Portal
   - Submit a case: "Enable Cortex AI features"
   - Include your account identifier

2. **Provide Account Details**:
   - Account identifier (e.g., `xy12345.us-east-1`)
   - Business justification for AI features
   - Expected usage patterns

### Step 2: Verify Cortex Functions
Once enabled, test these functions:

```sql
-- Test semantic search
SELECT CORTEX_SEARCH('diabetes patients', columns => ARRAY_CONSTRUCT('Hx_Diabetes', 'Age', 'GENDER'))
FROM ASN LIMIT 5;

-- Test text completion
SELECT CORTEX_COMPLETE('Find patients with', max_tokens => 10);

-- Test embeddings
SELECT CORTEX_EMBED('diabetes', model => 'e5-base-v2');
```

### Step 3: Optimize Your Data
For best Cortex AI performance:

```sql
-- Ensure text columns are properly formatted
ALTER TABLE ASN MODIFY COLUMN Hx_Diabetes VARCHAR(50);
ALTER TABLE ASN MODIFY COLUMN Sx_Vision VARCHAR(100);

-- Add appropriate indexes
CREATE INDEX idx_diabetes ON ASN(Hx_Diabetes);
CREATE INDEX idx_vision ON ASN(Sx_Vision);
```

## 🎯 Usage Examples

### Basic Semantic Search
```sql
-- Find patients with diabetes
SELECT *,
       CORTEX_SEARCH('diabetes patients', columns => ARRAY_CONSTRUCT('Hx_Diabetes', 'Age', 'GENDER')) as relevance
FROM ASN 
WHERE CORTEX_SEARCH('diabetes patients', columns => ARRAY_CONSTRUCT('Hx_Diabetes', 'Age', 'GENDER')) > 0.5
ORDER BY relevance DESC;
```

### Complex Medical Queries
```sql
-- Find elderly diabetic patients with vision problems
SELECT *,
       CORTEX_SEARCH('elderly diabetic patients with vision problems', 
                    columns => ARRAY_CONSTRUCT('Hx_Diabetes', 'Sx_Vision', 'Age', 'GENDER')) as relevance
FROM ASN 
WHERE Age > 65
  AND CORTEX_SEARCH('elderly diabetic patients with vision problems', 
                   columns => ARRAY_CONSTRUCT('Hx_Diabetes', 'Sx_Vision', 'Age', 'GENDER')) > 0.3
ORDER BY relevance DESC;
```

### Hybrid Search (AI + Traditional)
```sql
-- Combine Cortex AI with traditional filters
SELECT *,
       CORTEX_SEARCH('hypertension medication', columns => ARRAY_CONSTRUCT('Hx_Hypertension', 'Med_*')) as ai_score
FROM ASN 
WHERE GENDER = 'Female'
  AND Age > 50
  AND CORTEX_SEARCH('hypertension medication', columns => ARRAY_CONSTRUCT('Hx_Hypertension', 'Med_*')) > 0.2
ORDER BY ai_score DESC;
```

## 💰 Cost Considerations

### Cortex AI Pricing
- **Semantic Search**: Included in compute costs
- **Text Completion**: Pay-per-token (very low cost)
- **Embeddings**: Pay-per-embedding (minimal cost)

### Cost Optimization
```sql
-- Limit search scope to reduce costs
SELECT * FROM ASN 
WHERE CORTEX_SEARCH('query', columns => ARRAY_CONSTRUCT('specific_columns')) > 0.1
LIMIT 1000;

-- Use appropriate warehouse size
ALTER WAREHOUSE your_warehouse SET WAREHOUSE_SIZE = 'SMALL';
```

## 🔒 Security & Compliance

### Data Privacy
- **No data leaves Snowflake**: All processing happens within Snowflake
- **Encryption**: Data encrypted at rest and in transit
- **Access controls**: Standard Snowflake security applies

### Compliance
- **HIPAA compliant**: Suitable for medical data
- **SOC 2 Type II**: Certified security standards
- **GDPR ready**: Data residency controls

## 🚨 Troubleshooting

### Common Issues

1. **"CORTEX_SEARCH function not found"**
   - Solution: Contact Snowflake support to enable Cortex AI

2. **"Insufficient privileges"**
   - Solution: Grant appropriate privileges to your role

3. **"Function not available in region"**
   - Solution: Check Cortex AI availability in your region

4. **Poor search results**
   - Solution: Optimize column selection and data quality

### Performance Tips

1. **Limit search columns**: Don't search all 1500 columns
2. **Use appropriate thresholds**: Start with 0.1-0.3 for relevance
3. **Index important columns**: Add indexes for frequently searched columns
4. **Optimize warehouse size**: Use appropriate compute resources

## 📞 Support

### Getting Help
1. **Snowflake Documentation**: [Cortex AI Guide](https://docs.snowflake.com/en/user-guide/cortex)
2. **Snowflake Support**: Submit cases through support portal
3. **Community**: Snowflake Community forums

### Contact Information
- **Snowflake Support**: support.snowflake.com
- **Account Team**: Your Snowflake account representative
- **Documentation**: docs.snowflake.com

---

**Note**: Cortex AI availability and features may vary by region and account type. Contact your Snowflake representative for specific details about your account. 