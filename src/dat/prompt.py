DAT_PROMPT = """You are an evaluator assessing the retrieval effectiveness of dense
retrieval ( Cosine Distance ) and BM25 retrieval for finding the
correct answer.
## Task :
Given a question and two top1 search results ( one from dense retrieval ,
one from BM25 retrieval ) , score each retrieval method from **0 to 5**
based on whether the correct answer is likely to appear in top2 , top3 , etc.
### ** Scoring Criteria :**
1. ** Direct hit --> 5 points **
- If the retrieved document directly answers the question , assign **5 points **.
2. ** Good wrong result ( High likelihood correct answer is nearby ) --> 3 -4 points **
- If the top1 result is ** conceptually close ** to the correct answer (e. g.,
 mentions relevant entities , related events , partial answer), it indicates the search method is in the right direction.
- Give **4** if it 's very close , **3** if somewhat close.
3. ** Bad wrong result ( Low likelihood correct answer is nearby ) --> 1 -2 points **
- If the top1 result is ** loosely related but misleading ** ( e.g.,shares keywords but changes context ),
correct answers might not be in top2 , top3.
- Give **2** if there 's a small chance correct answers are nearby, **1** if unlikely.
4. ** Completely off - track --> 0 points **
- If the result is ** totally unrelated ** , it means the retrieval method is failing .
---
### ** Given Data :**
- ** Question :** "{question}"
- ** dense retrieval Top1 Result :** "{vector_reference}"
- ** BM25 retrieval Top1 Result :** "{bm25_reference}"
---
### ** Output Format :**
Return two integers separated by a space :
- ** First number :** dense retrieval score.
- ** Second number :** BM25 retrieval score.
- Example output : 3 4
( Vector : 3 , BM25 : 4)
** Do not output any other text .**
"""
