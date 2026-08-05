"""
memory/emr_memory.py

MEMORY SYSTEM -- The long-term memory of DrHouse.

Key concepts:
- Long-term memory is implemented as a vector store (ChromaDB)
- Each patient record is chunked, embedded, and indexed
- At query time, we retrieve the k most semantically relevant chunks
- This is RAG (Retrieval-Augmented Generation) applied to EMR data
- The agent never "knows" patient data -- it retrieves it on demand
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os



# ── SYNTHETIC EMR DATA ────────────────────────────────────────────────────────
# In production: these would come from the real EMR system via a secure API
# For the demo: we use realistic synthetic patient notes

SYNTHETIC_EMR_NOTES = [
    {
        "patient_id": "P001",
        "patient_name": "John Smith",
        "date": "2024-03-15",
        "type": "physician_note",
        "content": """
        Patient: John Smith, DOB: 1965-04-12
        Visit Date: 2024-03-15
        Physician: Dr. House

        Chief Complaint: Follow-up for Crohn's disease management.

        History: Patient has a 15-year history of Crohn's disease affecting the
        terminal ileum. Currently on Adalimumab 40mg biweekly. Last colonoscopy
        12 months ago showed mild mucosal inflammation at terminal ileum.

        Current Symptoms: Reports 2-3 loose stools per day, mild cramping post-meals.
        No blood in stool. Fatigue level 4/10.

        Assessment: Crohn's disease, mild-moderate activity. Adalimumab therapy
        showing partial response.

        Plan:
        1. Continue Adalimumab 40mg biweekly
        2. Add Budesonide 9mg daily for 8 weeks
        3. Repeat fecal calprotectin in 6 weeks
        4. Schedule colonoscopy in 6 months
        5. Dietary consult referral
        """
    },
    {
        "patient_id": "P001",
        "patient_name": "John Smith",
        "date": "2024-06-20",
        "type": "patient_email",
        "content": """
        From: John Smith <jsmith@email.com>
        Date: 2024-06-20
        Subject: Medication Side Effects

        Dear Dr. House,

        I have been experiencing significant nausea since starting the Budesonide
        last month. It is affecting my ability to eat and I have lost 3kg in the
        past 4 weeks. Should I continue taking it?

        Also, my calprotectin results came back at 450 ug/g. Is that concerning?

        John Smith
        """
    },
    {
        "patient_id": "P001",
        "patient_name": "John Smith",
        "date": "2024-06-21",
        "type": "physician_response",
        "content": """
        From: Dr. House <dhouse@pacificdigestive.ca>
        Date: 2024-06-21

        Dear John,

        Thank you for reaching out. The nausea you describe is a known side effect
        of Budesonide. Given the weight loss, please reduce to 6mg daily for 2 weeks,
        then 3mg for 2 weeks before stopping.

        A calprotectin of 450 is elevated (normal <50) and suggests ongoing
        inflammation. This confirms we need to reassess your Adalimumab dose.

        Please book an urgent appointment this week.

        Dr. House
        """
    },
    {
        "patient_id": "P002",
        "patient_name": "Sarah Chen",
        "date": "2024-01-10",
        "type": "physician_note",
        "content": """
        Patient: Sarah Chen, DOB: 1980-09-23
        Visit Date: 2024-01-10
        Physician: Dr. House

        Chief Complaint: New patient consultation for recurrent abdominal pain.

        History: 3-year history of intermittent abdominal pain, bloating, and
        alternating bowel habits. Previous workup at another centre: normal
        colonoscopy 2 years ago, normal blood work.

        Symptoms: Pain worse after meals, relieved partially by bowel movement.
        Bloating 7/10. No weight loss, no blood in stool, no nocturnal symptoms.

        Assessment: Irritable Bowel Syndrome (IBS), mixed type.

        Plan:
        1. Low FODMAP diet trial for 6 weeks
        2. Peppermint oil capsules 0.2mL TID
        3. Cognitive behavioural therapy referral for gut-brain axis
        4. Return in 6 weeks for follow-up
        5. If no improvement: consider Rifaximin trial
        """
    },
    {
        "patient_id": "P003",
        "patient_name": "Robert Martinez",
        "date": "2023-11-05",
        "type": "physician_note",
        "content": """
        Patient: Robert Martinez, DOB: 1958-12-01
        Visit Date: 2023-11-05
        Physician: Dr. House

        Chief Complaint: Surveillance colonoscopy results discussion.

        History: History of colorectal polyps. This is 3rd surveillance colonoscopy.

        Findings: 2 tubular adenomas <5mm in sigmoid colon. Removed successfully.
        No high-risk features. No serrated polyps.

        Assessment: Low-risk adenomatous polyps, completely resected.

        Plan:
        1. Next surveillance colonoscopy in 3 years (2026)
        2. Increase dietary fibre
        3. Aspirin 81mg daily discussed - patient to discuss with GP
        4. Remind patient of next colonoscopy date in 2026
        """
    },
]


class EMRMemorySystem:
    """
    Long-term memory system for DrHouse.

    Design rationale:
    This is a RAG-based memory system. Patient EMR records are:
    1. Chunked into smaller pieces (so we don't exceed context limits)
    2. Embedded into vectors (semantic representation)
    3. Stored in ChromaDB (persistent vector store)
    4. Retrieved at query time by semantic similarity

    The agent never loads all patient data -- it retrieves only what's
    relevant to the current query. This is how we handle 10 years of EMR data
    without exceeding the LLM context window.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " "]
        )
        self.vectorstore = None

    def build_memory(self, emr_records: list) -> None:
        """
        Index all EMR records into the vector store.

        In production: run this once when setting up, then incrementally
        update as new notes are added to the EMR.
        """
        documents = []

        for record in emr_records:
            # Split long notes into chunks
            chunks = self.text_splitter.split_text(record["content"])

            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "patient_id": record["patient_id"],
                        "patient_name": record["patient_name"],
                        "date": record["date"],
                        "type": record["type"],
                        "chunk_index": i,
                    }
                )
                documents.append(doc)

        print(f"Indexing {len(documents)} chunks from {len(emr_records)} records...")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print("Memory system ready.")

    def retrieve(self, query: str, patient_id: str = None, k: int = 4) -> list:
        """
        Retrieve relevant EMR context for a given query.

        Args:
            query: The current patient message or question
            patient_id: If provided, filter to this patient only
            k: Number of chunks to retrieve

        Returns:
            List of relevant Document chunks with metadata

        Design rationale:
        We use semantic similarity search -- the query is embedded into
        the same vector space as the EMR chunks. The k nearest neighbours
        (most semantically similar chunks) are returned. This means
        "stomach pain after eating" will retrieve notes about abdominal
        pain even if the exact words don't match.
        """
        if self.vectorstore is None:
            raise ValueError("Memory not built yet. Call build_memory() first.")

        # Filter by patient if specified
        filter_dict = {"patient_id": patient_id} if patient_id else None

        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        return results

    def get_patient_summary(self, patient_id: str) -> str:
        """
        Get a summary of all retrieved context for a patient.
        Formats retrieved chunks into a readable context string for the LLM.
        """
        results = self.retrieve(
            query="patient history diagnosis medication treatment plan",
            patient_id=patient_id,
            k=6
        )

        if not results:
            return f"No records found for patient {patient_id}."

        context_parts = []
        for doc in results:
            context_parts.append(
                f"[{doc.metadata['date']} - {doc.metadata['type']}]\n"
                f"{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    # Test the memory system
    memory = EMRMemorySystem(persist_directory="./chroma_db")
    memory.build_memory(SYNTHETIC_EMR_NOTES)

    print("\n--- TEST RETRIEVAL ---")
    print("Query: 'patient having nausea from medication'")
    results = memory.retrieve(
        query="patient having nausea from medication",
        patient_id="P001"
    )
    for r in results:
        print(f"\n[{r.metadata['date']}] {r.page_content[:200]}...")
