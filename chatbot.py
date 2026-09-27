import os
import sys

# Ensure UTF-8 output handling for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file (local) "
        "or your hosting provider's environment variables (Render, etc.)."
    )

# Document text: Nand2Tetris Practice Assignment (Morris Mano Based Circuits & ALU)
DUMMY_TEXT = """
Nand2Tetris Practice Assignment
Morris Mano Based Arithmetic Circuits, Multiplexers and ALU
K.R. Mangalam University — School of Engineering and Technology
Department of Computer Science and Engineering
Subject: Computer Organization and Architecture
Practical: Nand2Tetris HDL Practice | Architecture: Morris Mano Based ALU
Word Size: 16-bit | Maximum Marks: 30

Table of Contents:
1. Part A — Multiplexers (Mux, Mux16, Mux4Way, Mux4Way16)
2. Part B — Adders (Half Adder, Full Adder)
3. Part C — 16-bit Addition (Add16)
4. Part D — Subtraction (Sub16, AddSub16)
5. Part E — Increment / Decrement (Inc16, Dec16)
6. Part F — Morris Mano Arithmetic Circuit + Control Table
7. Part G — Morris Mano Logic Circuit + Control Table
8. Part H — Complete Morris Mano ALU + Final Control Table
9. Part N — Short Practice Questions (Detailed Answers)
10. Notes on Required Test Cases (Parts J and K)

Part A — Multiplexers:
Mux is built at the gate level from And/Or/Not: out = (a AND NOT sel) OR (b AND sel).
Mux16 applies Mux bitwise across 16 lines.
Mux4Way and Mux4Way16 are each built from three 2:1 multiplexers (two first-stage selecting by sel[0], one second-stage selecting by sel[1]).
Mux4Way selection:
sel[1] sel[0] -> out
0 0 -> a
0 1 -> b
1 0 -> c
1 1 -> d

Part B — Adders:
HalfAdder computes sum = a XOR b and carry = a AND b.
FullAdder is built from two Half Adders and one OR gate, adding a, b, and an incoming carry c:
HalfAdder(a=a, b=b, sum=sum1, carry=carry1);
HalfAdder(a=sum1, b=c, sum=sum, carry=carry2);
Or(a=carry1, b=carry2, out=carry);

Part C — 16-bit Addition (Add16):
Bit 0 uses a Half Adder (no carry-in); bits 1–15 use Full Adders chained by carry.
The final carry-out is discarded, so the sum wraps modulo 2^16 (e.g. 65535 + 1 = 0).

Part D — Subtraction:
Sub16 implements A - B = A + (NOT B) + 1: every bit of B is inverted, and the least-significant Full Adder carry-in is tied to constant 'true' to supply the '+1'.
AddSub16 wraps both Add16 and Sub16 behind a Mux16 selected by the 'sub' control line (sub=0 -> A+B, sub=1 -> A-B).

Part E — Increment and Decrement:
Inc16 reuses Add16 with the constant 0000000000000001 (b[0]=true, b[1..15]=false).
Dec16 reuses Add16 with the constant 1111111111111111 (b[0..15]=true), representing -1 in two's complement, so in + (-1) = in - 1.

Part F — Morris Mano Arithmetic Circuit:
A 4-way multiplexer selects the second adder operand Y from {B, NOT B, 0, all-ones} based on (S1, S0); a 16-bit Full-Adder chain then computes F = A + Y + Cin.
Control Table — ManoArithmetic:
- S1=0, S0=0, Y=B, Cin=0 -> F = A + B (Add)
- S1=0, S0=0, Y=B, Cin=1 -> F = A + B + 1 (Add with carry)
- S1=0, S0=1, Y=NOT B, Cin=0 -> F = A + NOT B (Subtract with borrow)
- S1=0, S0=1, Y=NOT B, Cin=1 -> F = A + NOT B + 1 = A - B (Subtract)
- S1=1, S0=0, Y=0, Cin=0 -> F = A (Transfer A)
- S1=1, S0=0, Y=0, Cin=1 -> F = A + 1 (Increment)
- S1=1, S0=1, Y=all-ones (-1), Cin=0 -> F = A - 1 (Decrement)
- S1=1, S0=1, Y=all-ones (-1), Cin=1 -> F = A (Transfer A duplicate)

Part G — Morris Mano Logic Circuit:
And16, Or16, a bitwise Xor bank, and Not16 are computed in parallel; a Mux4Way16 selects the active result using (L1, L0).
Control Table:
- L1=0, L0=0: AND (F = A AND B)
- L1=0, L0=1: OR (F = A OR B)
- L1=1, L0=0: XOR (F = A XOR B)
- L1=1, L0=1: NOT (F = NOT A)

Part H — Complete Morris Mano ALU:
S[2] is shared between ManoArithmetic and ManoLogic; M selects which sub-circuit result reaches F (M=0 arithmetic mode, M=1 logic mode).
Flags:
- Zero flag zr = 1 when every bit of F is 0 (built from two Or8Way chips over F[0..7] and F[8..15], combined with Or, then inverted with Not).
- Negative flag ng = F[15], the sign bit under 16-bit two's-complement representation (1 for negative, 0 for non-negative).
Final Control Table:
- Func 1: M=0, S1=1, S0=0, Cin=0 -> Transfer (F = A)
- Func 2: M=0, S1=1, S0=0, Cin=1 -> Increment (F = A + 1)
- Func 3: M=0, S1=1, S0=1, Cin=0 -> Decrement (F = A - 1)
- Func 4: M=0, S1=0, S0=0, Cin=0 -> Add (F = A + B)
- Func 5: M=0, S1=0, S0=1, Cin=1 -> Subtract (F = A - B)
- Func 6: M=1, S1=0, S0=0, Cin=X -> AND (F = A AND B)
- Func 7: M=1, S1=0, S0=1, Cin=X -> OR (F = A OR B)
- Func 8: M=1, S1=1, S0=0, Cin=X -> XOR (F = A XOR B)
- Func 9: M=1, S1=1, S0=1, Cin=X -> NOT (F = NOT A)

Part N — Short Practice Questions & Answers:
1. What is the function of a 2:1 MUX?
A 2:1 multiplexer selects one of two 1-bit input signals (a or b) to route to a single output, based on the value of a select line sel. When sel = 0 the output equals a; when sel = 1 the output equals b. It is the hardware equivalent of an if/else switch for a single bit.

2. How many 2:1 MUXes are required to construct a 4:1 MUX?
Three. Two first-level 2:1 MUXes narrow the four inputs (a, b, c, d) down to two intermediate signals using low-order select bit sel[0], and a third 2:1 MUX picks between those two intermediate results using high-order select bit sel[1].

3. What is the difference between a Half Adder and a Full Adder?
A Half Adder adds exactly two single bits (a and b) and produces a sum and carry output — it cannot accept an incoming carry. A Full Adder adds three bits (a, b, and incoming carry c), producing sum and carry outputs, suitable for chaining into multi-bit adders.

4. What is the purpose of the carry input in a Full Adder?
The carry input lets a Full Adder receive the carry-out generated by the adder stage handling the next-lower-order bit, allowing multiple Full Adders to be chained into a ripple-carry adder that propagates carries from LSB toward MSB.

5. Write the equation used for two's-complement subtraction.
A - B = A + (NOT B) + 1. Subtraction is performed by adding the minuend A to the one's complement of the subtrahend B, plus 1.

6. Why is +1 required after complementing the subtracted number?
Complementing every bit of B (one's complement, NOT B) computes -B - 1, not -B. Adding 1 afterward corrects this off-by-one and yields the true two's-complement negation -B = NOT B + 1.

7. What operation is performed by an incrementer?
An incrementer (Inc16) adds 1 to a 16-bit input value, producing out = in + 1. Built by feeding constant 0000000000000001 as the second operand of a 16-bit adder.

8. What operation is performed by a decrementer?
A decrementer (Dec16) subtracts 1 from a 16-bit input value, producing out = in - 1. Built by feeding constant 1111111111111111 (-1 in two's complement) as the second operand of a 16-bit adder.

9. What is the purpose of the arithmetic circuit in an ALU?
The arithmetic circuit (ManoArithmetic) performs numeric operations such as addition, subtraction, increment, decrement, and transfer by combining an input-selection stage (B, NOT B, 0, all-ones) with a chain of full adders and carry-in.

10. What is the purpose of the logic circuit in an ALU?
The logic circuit (ManoLogic) performs bitwise Boolean operations — AND, OR, XOR, NOT — in parallel, using a multiplexer controlled by L1 and L0 to select the active result.

11. What is the purpose of the arithmetic/logic selection MUX?
It is the final-stage multiplexer (controlled by M) deciding whether the ALU output F comes from the arithmetic circuit (M=0) or logic circuit (M=1).

12. How is the zero flag generated?
The zero flag zr is generated by OR-ing together every bit of the final output F using two Or8Way chips (for bits 0-7 and 8-15) and an Or gate, then inverting with a Not gate so that zr = 1 exactly when all 16 bits are 0.

13. Which bit is used for the negative flag in a 16-bit two's-complement number?
Bit 15 (most significant bit, F[15]) is used directly as the negative flag ng: 1 indicates negative, 0 indicates non-negative.
"""

def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_text(DUMMY_TEXT)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def main():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.2,
    )

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    print("=" * 60)
    print("[*] RAG Chatbot ready (dummy document loaded).")
    print(f"[*] Using Model: {GROQ_MODEL}")
    print("[*] Type 'exit' or 'quit' to end the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        docs = retriever.invoke(user_input)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = (
            "greet user, and try to answer the question using only the context below. "
            f"Context:\n{context}\n\nQuestion: {user_input}"
        )

        try:
            response = llm.invoke(prompt)
            print(f"Bot: {response.content}\n")
        except Exception as e:
            print(f"Error querying model: {e}\n")

if __name__ == "__main__":
    main()