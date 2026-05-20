# Trust Scoring Notes

TrustDoc AI estimates answer reliability using retrieval evidence, not model confidence.

## Signals Used

- Top retrieval score: whether the closest chunk strongly matches the question
- Average score of the top chunks: whether support is consistent across retrieved evidence
- Question/evidence overlap: whether important question terms appear in the evidence
- Number of supporting chunks: whether the answer has multiple pieces of support
- Number of sources: whether more than one document supports the answer
- Conflict warnings: whether retrieved chunks contain possibly inconsistent language
- Knowledge gaps: whether the system found weak or missing evidence

## Refusal Guardrail

The app refuses to answer when evidence is too weak. This avoids unsupported responses and makes the chatbot safer for real-world document use.

Example refusal:

```text
I could not find enough support in the uploaded documents to answer this reliably.
Try uploading a more relevant document or asking a narrower question.
```

## Important Note

The trust score is a practical heuristic for a portfolio project. It is not a formal probability and should not be treated as legal, medical, financial, or compliance advice.

