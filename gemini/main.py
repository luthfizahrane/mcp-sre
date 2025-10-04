#main.py
import os
import random
import uuid
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from pydantic import BaseModel, Field, ValidationError
from supabase import create_client, Client
import asyncpg
from pgvector.asyncpg import register_vector
from db import _get_pg_connection, close_pg_pool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Import semua modul dari kode MCP Anda
from agentic_new import *
from supabase_vector_db import SupabaseVectorDb

from graph_service import graph_service
from PyPDF2 import PdfReader
from io import BytesIO
import traceback
import logging 
from fastapi.logger import logger as fastapi_logger
import json
import uvicorn

from markitdown import MarkItDown
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Integrate with FastAPI's logger
fastapi_logger.handlers = logger.handlers
fastapi_logger.setLevel(logger.level)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

print(f"Using Google API Key: {GOOGLE_API_KEY[:5]}...")  # Debugging

model_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
llm_kg = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.5-pro", temperature=0.2) 
embeddings_for_retrieval = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
embeddings_kg_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")


# --- KELAS PYDANTIC UNTUK KG (DARI kg_processor.py) ---
class ArticleContentSections(BaseModel):
    background: Optional[str] = Field(None, description="Isi latar belakang penelitian.")
    methodology: Optional[str] = Field(None, description="Metodologi penelitian yang digunakan.")
    purpose: Optional[str] = Field(None, description="Tujuan penelitian ini.")
    future_research: Optional[str] = Field(None, description="Rekomendasi untuk penelitian lebih lanjut.")
    research_gap: Optional[str] = Field(None, description="Kesenjangan penelitian atau batasan yang diidentifikasi.")

class ArticleMainNode(BaseModel):
    id: str = Field(description="ID node (e.g., doc_<UUID>)")
    label: str = Field(description="Label visual node (e.g., 'Artikel')")
    title: Optional[str] = Field(None, description="Judul artikel")
    att_goal: Optional[str] = Field(None, description="Tujuan penelitian (dari purpose)")
    att_method: Optional[str] = Field(None, description="Metodologi penelitian (dari methodology)")
    att_background: Optional[str] = Field(None, description="Latar belakang penelitian (dari background)")
    att_future: Optional[str] = Field(None, description="Saran penelitian lanjutan (dari future_research)")
    att_gaps: Optional[str] = Field(None, description="Gap penelitian (dari research_gap)")
    type: str = Field(description="Tipe node (e.g., 'article' dari skema JS)")
    content: str = Field(description="Ringkasan singkat keseluruhan dokumen.")

class Relation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID unik edge")
    fromId: str = Field(description="ID node sumber")
    toId: str = Field(description="ID node target")
    relation: str = Field(description="Tipe relasi")
    label: Optional[str] = Field(None, description="Label visual edge")

class RelationVerificationOutput(BaseModel):
    type: Optional[str] = Field(None, description="Tipe relasi jika terdeteksi, atau null")
    context: str = Field(description="Penjelasan mengapa relasi ada atau tidak ada.")

class KnowledgeGraphOutput(BaseModel):
    article_node: ArticleMainNode = Field(description="Node utama untuk artikel yang sedang diproses.")


# --- PROMPT EKSTRAKSI NODE KG (DARI kg_processor.py) ---
KG_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["retrieved_context", "document_id", "document_title", "existing_nodes_json", "format_instruction"],
    template="""
    Anda adalah agen ekstraksi informasi ahli yang membangun knowledge graph untuk penelitian akademik.
    
    TUGAS UTAMA:
    Ekstrak informasi terstruktur untuk node artikel BARU berdasarkan KONTEKS yang DIBERIKAN.
    Output Anda HARUS berupa data JSON yang valid sesuai skema yang diberikan, BUKAN definisi skemanya.
    Pastikan untuk mengekstrak secara eksplisit:
    - Latar Belakang Penelitian (cari frasa seperti 'latar belakang', 'pendahuluan', atau konteks awal; jika tidak ditemukan, lakukan inferensi dari bagian pengantar atau konteks awal lainnya sebelum menyimpulkan 'Tidak ditemukan')
    - Metodologi Penelitian (cari frasa seperti 'metode', 'pendekatan', 'prosedur'; jika tidak ditemukan, lakukan inferensi dari deskripsi proses atau langkah-langkah penelitian sebelum menyimpulkan 'Tidak ditemukan')
    - Tujuan Penelitian (cari frasa seperti 'tujuan', 'sasaran', 'maksud'; jika tidak ditemukan, lakukan inferensi dari implikasi latar belakang atau hasil yang diharapkan sebelum menyimpulkan 'Tidak ditemukan')
    - Penelitian Lanjutan (cari frasa seperti 'penelitian lanjutan', 'rekomendasi', 'studi lebih lanjut'; jika tidak ditemukan, lakukan inferensi dari kesimpulan, pembahasan, atau saran implisit terkait ruang lingkup penelitian sebelum menyimpulkan 'Tidak ditemukan')
    - Gap Penelitian (cari frasa seperti 'keterbatasan', 'kesenjangan', 'tantangan'; jika tidak ditemukan, lakukan inferensi dari kesimpulan, pembahasan, atau kelemahan terkait ruang lingkup penelitian sebelum menyimpulkan 'Tidak ditemukan')
    
    Jika informasi tidak ditemukan setelah upaya inferensi, isi dengan 'Tidak ditemukan' hanya jika tidak ada petunjuk sama sekali.

    KONTEKS ARTIKEL BARU:
    Judul: {document_title}
    ID: {document_id}
    Konten: {retrieved_context}
    
    ARTIKEL YANG SUDAH ADA (untuk referensi, bukan untuk membuat relasi otomatis di sini):
    {existing_nodes_json}
    
    INSTRUKSI EKSTRAKSI NODE ARTIKEL BARU:
    Untuk 'article_node' yang sedang diproses, ekstrak:
    - ID: 'doc_{document_id}'
    - Label: 'Artikel' (tetap)
    - Judul: {document_title}
    - Att_goal: Tujuan penelitian (jika tidak ada, cari implikasi dari latar belakang atau metode)
    - Att_method: Metodologi penelitian (jika tidak ada, cari deskripsi proses atau eksperimen)
    - Att_background: Latar belakang penelitian (jika tidak ada, gunakan bagian pendahuluan)
    - Att_future: Saran untuk penelitian lanjutan (jika tidak ada, cari saran implisit)
    - Att_gaps: Kesenjangan atau keterbatasan yang diidentifikasi (jika tidak ada, cari kelemahan metode atau data)
    - Type: 'article' (tetap)
    - Content: Ringkasan komprehensif maksimal 500 kata dari seluruh dokumen.
    Jika suatu detail tidak ditemukan meskipun ada petunjuk konteks, lakukan inferensi logis berdasarkan teks yang tersedia.
    
    CONTOH OUTPUT DATA (Ikuti format ini persis):
    ```json
    {{
      "article_node": {{
        "id": "doc_example_id_123",
        "label": "Artikel",
        "title": "Contoh Judul Artikel",
        "att_goal": "Tujuan contoh",
        "att_method": "Metodologi contoh",
        "att_background": "Latar belakang contoh.",
        "att_future": "Saran penelitian contoh.",
        "att_gaps": "Gap penelitian contoh.",
        "type": "article",
        "content": "Ini adalah ringkasan contoh dari artikel tersebut."
        
      }}
    }}
    ```

    Output harus JSON murni sesuai skema:
    {format_instruction}
    """,
)

# Format instruksi untuk verifikasi relasi
FORMAT_INSTRUCTION_REL_VERIFY = json.dumps(RelationVerificationOutput.model_json_schema(), indent=2)

# Prompts untuk Verifikasi Relasi Per Aspek
RELATION_PROMPTS_MAP = {
    "SERUPA_LATAR_BELAKANG": PromptTemplate(
        input_variables=["article1_id", "article1_details", "article2_id", "article2_details", "format_instruction"],
        template=f"""
        Anda adalah agen pembuat relasi knowledge graph.
        Tugas Anda adalah menentukan apakah LATAR BELAKANG kedua artikel ini cukup mirip dan signifikan untuk membuat relasi 'SERUPA_LATAR_BELAKANG'.
        Output Anda HARUS berupa JSON yang valid.
        
        Artikel 1 (ID: {{article1_id}}) Latar Belakang: {{article1_details.background}}
        Artikel 2 (ID: {{article2_id}}) Latar Belakang: {{article2_details.background}}

        Output JSON harus sesuai skema Pydantic berikut:
        {FORMAT_INSTRUCTION_REL_VERIFY}
        """,
    ),
    "SERUPA_TUJUAN": PromptTemplate(
        input_variables=["article1_id", "article1_details", "article2_id", "article2_details", "format_instruction"],
        template=f"""
        Anda adalah agen pembuat relasi knowledge graph.
        Tugas Anda adalah menentukan apakah TUJUAN kedua artikel ini cukup mirip dan signifikan untuk membuat relasi 'SERUPA_TUJUAN'.
        Output Anda HARUS berupa JSON yang valid.

        Artikel 1 (ID: {{article1_id}}) Tujuan: {{article1_details.purpose}}
        Artikel 2 (ID: {{article2_id}}) Tujuan: {{article2_details.purpose}}

        Output JSON harus sesuai skema Pydantic berikut:
        {FORMAT_INSTRUCTION_REL_VERIFY}
        """,
    ),
    "SERUPA_METODOLOGI": PromptTemplate(
        input_variables=["article1_id", "article1_details", "article2_id", "article2_details", "format_instruction"],
        template=f"""
        Anda adalah agen pembuat relasi knowledge graph.
        Tugas Anda adalah menentukan apakah METODOLOGI kedua artikel ini cukup mirip dan signifikan untuk membuat relasi 'SERUPA_METODOLOGI'.
        Output Anda HARUS berupa JSON yang valid.

        Artikel 1 (ID: {{article1_id}}) Metodologi: {{article1_details.methodology}}
        Artikel 2 (ID: {{article2_id}}) Metodologi: {{article2_details.methodology}}

        Output JSON harus sesuai skema Pydantic berikut:
        {FORMAT_INSTRUCTION_REL_VERIFY}
        """,
    ),
    "SERUPA_PENELITIAN_LANJUT": PromptTemplate(
        input_variables=["article1_id", "article1_details", "article2_id", "article2_details", "format_instruction"],
        template=f"""
        Anda adalah agen pembuat relasi knowledge graph.
        Tugas Anda adalah menentukan apakah REKOMENDASI PENELITIAN LANJUT kedua artikel ini cukup mirip dan signifikan untuk membuat relasi 'SERUPA_PENELITIAN_LANJUT'.
        Output Anda HARUS berupa JSON yang valid.

        Artikel 1 (ID: {{article1_id}}) Penelitian Lanjut: {{article1_details.future_research}}
        Artikel 2 (ID: {{article2_id}}) Penelitian Lanjut: {{article2_details.future_research}}

        Output JSON harus sesuai skema Pydantic berikut:
        {FORMAT_INSTRUCTION_REL_VERIFY}
        """,
    ),
    "SERUPA_GAP_PENELITIAN": PromptTemplate(
        input_variables=["article1_id", "article1_details", "article2_id", "article2_details", "format_instruction"],
        template=f"""
        Anda adalah agen pembuat relasi knowledge graph.
        Tugas Anda adalah menentukan apakah GAP PENELITIAN kedua artikel ini cukup mirip dan signifikan untuk membuat relasi 'SERUPA_GAP_PENELITIAN'.
        Output Anda HARUS berupa JSON yang valid.

        Artikel 1 (ID: {{article1_id}}) Gap Penelitian: {{article1_details.research_gap}}
        Artikel 2 (ID: {{article2_id}}) Gap Penelitian: {{article2_details.research_gap}}

        Output JSON harus sesuai skema Pydantic berikut:
        {FORMAT_INSTRUCTION_REL_VERIFY}
        """,
    ),
}
# Buat LLMChain untuk setiap prompt relasi secara global
RELATION_CHAINS = {
    rel_type: prompt | llm_kg
    for rel_type, prompt in RELATION_PROMPTS_MAP.items()
}
# --- UTILITAS DATABASE UNTUK KG (DARI kg_processor.py) ---
# Postgres connection is provided by `db._get_pg_connection()` which uses a connection pool.

async def _get_existing_nodes() -> List[Dict[str, Any]]:
    # Mengambil node yang sudah ada untuk konteks prompt relasi
    try:
        conn = await _get_pg_connection()
        # Query untuk mengambil semua data node yang dibutuhkan LLM
        nodes = await conn.fetch("SELECT id, title, att_goal, att_method, att_background, att_future, att_gaps, type, content FROM nodes_kg")
        await conn.close()
        return [dict(node) for node in nodes]
    except Exception as e:
        print(f"[KG_Utils] Error fetching existing nodes: {e}")
        return []
class ProcessPDFRequest(BaseModel):
    pdf_url: str
    session_id: str
    node_id: Optional[str] = None
    metadata: Optional[dict[str,str]] = {}

class GenerateEdgesRequest(BaseModel):
    all_nodes_data: List[Dict[str, Any]]

app = FastAPI(title="MCP Agentic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "documents"
TABLE_NAME = "documents"
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))

class SupabaseVectorWrapper:
    """Wrapper untuk SupabaseVectroDb agar kompatibel"""

    def __init__(self, table_name: str = "documents"):
        self.vector_db = SupabaseVectorDb(table_name=table_name)
        self.collection_name = table_name

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Wrapper untuk add_documents"""
        return self.vector_db.add_documents(
            texts=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def similarity_search(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None):
        return self.vector_db.similarity_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )
    
    def get_collection_info(self):
        """Wrapper get collection info"""
        return self.vector_db.get_collection_info()
    
    def delete_documents(self, ids: List[str]):
        """Wrapper untuk delete documents"""
        return self.vector_db.delete_documents(ids)
    
    async def health_check(self, collection_name: Optional[str] = None) -> bool:
        try:
            print(f"[SupabaseHealthCheck] Called for: {collection_name} vs actual: {self.collection_name}")

            if collection_name and collection_name != self.collection_name:
                print(f"[SupabaseHealthCheck] Collection mismatch!")
                return False

            test = self.vector_db.similarity_search("test", k=1)
            print(f"[SupabaseHealthCheck] Similarity search results: {len(test)}")
            return True
        except Exception as e:
            print(f"[SupabaseHealthCheck] ERROR: {e}")
            return False

def init_system():
    """Initialize semua modul"""
    print("Initializing system...")
    print("Initializing with supabase...")

    try:
        supabase_db = SupabaseVectorWrapper(table_name=TABLE_NAME)
        print("v supabase vector db initalized success")
    except Exception as e:
        print(f"X failed to initialize Supabase: {e}")
        raise
    
    mcp_orchestrator = MCPOrchestrator()
    mcp_orchestrator.register_provider(VectorDBMCPProvider(supabase_db, TABLE_NAME))
    mcp_orchestrator.register_provider(WebSearchMCPProvider())
    mcp_orchestrator.register_provider(GraphDBMCPProvider(graph_service))
    mcp_orchestrator.register_provider(GraphReasoningProvider(graph_service))

    
    perception_module = EnhancedPerceptionModule()
    reasoning_module = EnhancedReasoningModule(mcp_orchestrator)
    action_module = EnhancedActionModule()
    learning_module = LearningModule()

    print(f"Action module: {action_module}") 
    
    return {
        "vector_db": supabase_db,
        "mcp": mcp_orchestrator,
        "perception": perception_module,
        "reasoning": reasoning_module,
        "action": action_module,
        "learning": learning_module
    }

system = init_system()
print("System initialized successfully")

class ChatRequest(BaseModel):
    question: str
    session_id: str
    mode: Literal['general', 'single_node', 'multi_nodes'] = 'general'
    node_id: Optional[str] = None
    node_ids: Optional[List[str]] = None
    force_web: bool = False
    context_node_ids: Optional[List[str]] = None
    context_edge_ids: Optional[List[str]] = None
    context_article_ids: Optional[List[str]] = None

    class Config:
        extra = "forbid"

class ProcessTextRequest(BaseModel):
    text: str
    session_id: str
    metadata: Optional[dict] = None

class SuggestionRequest(BaseModel):
    query: str
    context: Optional[dict] = None
    suggestion_type: Literal["input", "followup"] = "input"  # or "followup"
    chat_history: Optional[List[Dict[str, str]]] = None

class FollowupRequest(BaseModel):
    lastMessage: str
    conversationHistory: Optional[List[dict]] = []
    context: Optional[dict] = None
    suggestion_type: str = "followup"

async def handle_chat(request: ChatRequest):
    """Endpoint utama untuk chat"""
    try:
        print(f"Received chat request: {request.model_dump()}")

        if not request.session_id:
            raise HTTPException(
                status_code=422,
                detail="session_id is required"
            )

        # Validasi mode
        valid_modes = ["general", "single_node", "multi_nodes"]
        if request.mode not in valid_modes:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid mode. Must be one of: {valid_modes}"
            )

        additional_context = {}
        context_node_ids = request.context_node_ids or []
        context_edge_ids = request.context_edge_ids or []
        context_article_ids = request.context_article_ids or []

        if (not context_node_ids and 
            not context_edge_ids and 
            not context_article_ids and 
            not request.node_id and 
            not request.node_ids):
            
            return {
                "success": True,
                "response": "Anda belum memilih node",
                "references": [],
                "usage_metadata": None,
                "metadata": {
                    "perception": {},
                    "reasoning": {},
                    "action": {"action_type": "no_context"}
                }
            }

        if context_node_ids or context_edge_ids:
            graph_context = await graph_service.get_graph_context(
                context_node_ids,
                context_edge_ids
            )
            additional_context["graph"] = graph_context

        # 1. Perception
        perception_data = await system["perception"].perceive(
            user_input = request.question, 
            context={
                **additional_context,
            })
        
        # 2. Reasoning
        reasoning_result_dict = await graph_service.reason(
            input = perception_data.user_input, 
            force_web = request.force_web,
            external_context={
                "node_ids": request.context_node_ids,
                "edge_ids": request.context_edge_ids,
            })
        
        reasoning_result = ReasoningResult(
            strategy=reasoning_result_dict.get("strategy", "hybrid"),
            confidence=reasoning_result_dict.get("confidence", 0.8),
            context_sources=reasoning_result_dict.get("context_sources", []),
            reasoning_chain=reasoning_result_dict.get("reasoning_chain", [])
        )

        external_filters = {}
        if request.context_node_ids is not None:
            external_filters["node_ids"] = request.context_node_ids
        if request.context_edge_ids is not None:
            external_filters["edge_ids"] = request.context_edge_ids

        # Filter BARU untuk VectorDB
        if context_article_ids:
            # PENTING: Kunci filternya harus 'article_id' karena itu yang disimpan di metadata Supabase
            external_filters["article_id"] = {"in": context_article_ids}
        
        # 3. Action
        unified_context = await system["mcp"].get_unified_context(
            query = request.question, 
            providers = reasoning_result.context_sources or [],
            external_filters=external_filters or None
        ) or {"providers": {}}

        # Pastikan unified_context tidak None
        if not unified_context:
            unified_context = {
                "query": request.question,
                "timestamp": datetime.now().isoformat(),
                "providers": {}
            }

        action_result = await system["action"].act(perception_data, reasoning_result, unified_context)
        
        # 4. Learning
        system["learning"].record_interaction(
            interaction_id = request.session_id,
            perception_data = perception_data,
            reasoning_result = reasoning_result,
            action_result = action_result,
            context_ids={
                "nodes": request.context_node_ids or [],
                "edges": request.context_edge_ids or [],
            }
        )
        
        return {
            "success": True,
            "response": action_result.response,
            "references": action_result.references,
            "usage_metadata": action_result.usage_metadata,
            "metadata": {
                "perception": perception_data.__dict__,
                "reasoning": reasoning_result.__dict__,
                "action": action_result.__dict__
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "message": "internal server error",
            "error": str(e),
            "type": type(e).__name__
        })

async def process_pdf_from_url(request: ProcessPDFRequest):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(request.pdf_url)
            response.raise_for_status()
            
            pdf_bytes = BytesIO(response.content)
            
            try:
                reader = PdfReader(pdf_bytes)
                full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                
                if not full_text.strip():
                    raise HTTPException(
                        status_code=422,
                        detail="PDF is empty or text extraction failed"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"PDF text extraction error: {str(e)}"
                )

        return await process_text_internal(full_text, request)
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download PDF: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

async def process_text_internal(text: str, request: ProcessPDFRequest) -> dict:
    try:
        # 1. Preprocessing teks
        processed_text = preprocess_indonesian_text(text)
        
        # 2. Split dokumen
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=lambda x: len(x.split())
        )
        
        # 3. Split dokumen menjadi chunks
        chunks = text_splitter.split_text(processed_text)
        node_id = request.node_id
        
        # 4. Menyiapkan metadata
        base_metadata = {
            "source_url": request.pdf_url or "",  # Convert None to empty string
            "session_id": request.session_id or "",
            "node_id": request.node_id or "",
            "article_id": request.metadata.get("article_id"),
            "language": "id",
            "title": request.metadata.get("title")
        }
        
        additional_metadata = request.metadata or {}
        sanitized_additional_metadata = {
            k: v if v is not None else "" for k, v in additional_metadata.items()
        }
        
        final_metadata = {**base_metadata, **sanitized_additional_metadata}
        
        metadatas = [final_metadata for _ in chunks]

        chunks_ids = [
            str(uuid.uuid4()) for _ in chunks
        ]

        # 5. Simpan ke VectorDB
        try:
            stored_ids = system["vector_db"].add_documents(
                documents=chunks,
                metadatas=metadatas,
                ids=chunks_ids
            )

            if not stored_ids or len(stored_ids) != len(chunks):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to store all document chunks"
                )
            print(f"✅ Stored {len(stored_ids)} chunks in VectorDB with IDs: {stored_ids[:3]}...")

            # TAHAP 4: PEMBUATAN NODE KG
            node_result = await generate_article_summary_node(
                full_document_content=processed_text,
                document_id=request.metadata.get("article_id"),
                document_title=request.metadata.get("title")
            )

            if not node_result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Gagal memproses node KG: {node_result.get('error')}"
                )

            # TAHAP 5: MEMPROSES DATA OUTPUT KG SESUAI STRUKTUR YANG DIMINTA USER
            generated_article_node = node_result.get("new_node_details", {})
            logger.info(f"DEBUG: generated_article_node before returning: {generated_article_node}")

            generated_article_node['att_url'] = request.pdf_url

            token_usage = generated_article_node.pop('token_usage', {})
        
            return {
                "success": True,
                "num_chunks": len(chunks),
                # "document_ids": [f"{request.session_id}_{i}" for i in range(len(chunks))]
                "document_ids": stored_ids,
                "node_id": node_id,
                "session_id": request.session_id,
                "generated_article_node": generated_article_node,
                "token_usage": token_usage
            }
        
        except Exception as e:
            print(f"error saving to Supabase: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
        
    except Exception as e:
        print(f"Error in text processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text processing error: {str(e)}"
        )
    
def preprocess_indonesian_text(text: str) -> str:
    """
    Preprocessing khusus teks Bahasa Indonesia.
    Handle karakter khusus, normalisasi, dll.
    """
    # 1. Normalisasi karakter khusus
    text = (
        text.replace("â€œ", '"')  # Kutipan curly
        .replace("â€ ", '"')
        .replace("â€™", "'")  # Apostrof
        .replace("â€˜", "'")
        .replace("â€”", "-")  # Dash
    )
    
    # 2. Koreksi singkatan umum
    replacements = {
        ' tdk ': ' tidak ',
        ' yg ': ' yang ',
        ' dgn ': ' dengan ',
        ' pd ': ' pada ',
        ' jg ': ' juga '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # 3. Hapus whitespace berlebihan
    text = ' '.join(text.split())
    
    return text

# --- FUNGSI AI: Generate Satu Node Ringkasan Artikel ---
async def generate_article_summary_node(full_document_content: str, document_id: str, document_title: str) -> Dict[str, Any]:
    print(f"Memulai pembuatan NODE ARTIKEL UTAMA untuk '{document_title}' (ID: {document_id})")

    conn_retrieval = None
    retrieved_context = ""
        
        # 1. RAG Retrieval (Get relevant chunks for context)
    try:
        conn_retrieval = await _get_pg_connection()
        query_text = f"Apa tujuan, metodologi, dan latar belakang utama dari dokumen berjudul: {document_title}? Ringkasan awal: {full_document_content[:500]}..."
        query_embedding = await embeddings_for_retrieval.aembed_query(query_text)
        
        # WARNING: Menggunakan table 'documents' dan metadata->>'article_id' sesuai repo
        # dan memastikan query filter hanya pada chunk dokumen ini
        retrieval_results = await conn_retrieval.fetch("""
            SELECT content AS chunk_text, embedding <-> $1 AS distance
            FROM documents 
            WHERE metadata->>'article_id' = $2 
            ORDER BY distance
            LIMIT 20;
        """, query_embedding, document_id)
        
        if retrieval_results:
            retrieved_context = "\n\n".join([r['chunk_text'] for r in retrieval_results])
            print(f"[KG_NODE] Berhasil mengambil {len(retrieval_results)} chunks untuk konteks.")
        else:
            retrieved_context = full_document_content[:2000]
            print("[KG_NODE] Tidak ada chunks yang diambil, menggunakan bagian awal dokumen penuh sebagai fallback.")

    except Exception as e:
        print(f"[KG_NODE] Error saat RAG retrieval: {e}")
        retrieved_context = full_document_content[:2000] 
    finally:
        if conn_retrieval: 
            await conn_retrieval.close()

    # 2. Get existing nodes for prompt context (untuk referensi, meski tidak dipakai node extraction)
    existing_article_nodes = await _get_existing_nodes()
    existing_nodes_json_for_prompt = json.dumps(existing_article_nodes, indent=2)
    format_instruction_main_node = json.dumps(KnowledgeGraphOutput.model_json_schema(), indent=2)
        
    # 3. LLM Call for Node Extraction
    try:
        kg_extraction_chain = KG_EXTRACTION_PROMPT | llm_kg 
        
        input_data_main_node = {
            "retrieved_context": retrieved_context,
            "document_id": document_id,
            "document_title": document_title,
            "existing_nodes_json": existing_nodes_json_for_prompt,
            "format_instruction": format_instruction_main_node
        }
        
        raw_kg_output_message = await kg_extraction_chain.ainvoke(input_data_main_node)
        raw_kg_output_str = raw_kg_output_message.content
        
        json_string_to_parse = raw_kg_output_str.strip().lstrip('```json').rstrip('```').strip()
        kg_output_dict = json.loads(json_string_to_parse)
        kg_output = KnowledgeGraphOutput.model_validate(kg_output_dict)
        main_article_node = kg_output.article_node

        # Skipping optional retry logic for missing fields for brevity.

        # 4. Save Node to nodes_kg
        conn_save = None
        try:
            conn_save = await _get_pg_connection()
            article_embedding_content = main_article_node.content if main_article_node.content else main_article_node.title
            article_node_embedding = await embeddings_kg_model.aembed_query(article_embedding_content)

            await conn_save.execute("""
                INSERT INTO nodes_kg (id, label, title, att_goal, att_method, att_background, att_future, att_gaps, type, content, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (id) DO NOTHING;
            """, 
                main_article_node.id,
                main_article_node.label,
                main_article_node.title,
                main_article_node.att_goal,
                main_article_node.att_method,
                main_article_node.att_background,
                main_article_node.att_future,
                main_article_node.att_gaps,
                main_article_node.type,
                main_article_node.content,
                article_node_embedding
            )
            print(f"[KG_NODE] Node Artikel Utama '{main_article_node.id}' berhasil disimpan.")
            
            return {
                "success": True, 
                "node_id": main_article_node.id, 
                "existing_nodes": existing_article_nodes,
                "new_node_details": main_article_node.model_dump() # Kembalikan detail node baru untuk langkah edges
            }

        except Exception as e:
            print(f"[KG_NODE] Gagal menyimpan Node Artikel Utama: {e}")
            return {"success": False, "error": str(e)}
        finally:
            if conn_save:
                await conn_save.close()

    except Exception as e:
        print(f"[KG_NODE] Error saat ekstraksi atau validasi Node: {e}")
        return {"success": False, "error": str(e)}


async def generate_edges_from_all_nodes(new_node: Dict[str, Any], existing_article_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Membuat edges antara node baru dengan semua node yang sudah ada.
    """
    default_token_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    final_relations_to_insert = []
    
    if not existing_article_nodes:
        print("Tidak ada artikel lain di database untuk dibandingkan relasi.")
        return {"edges": final_relations_to_insert, "token_usage": default_token_usage}

    # Model Validation
    try:
        main_article_node = ArticleMainNode.model_validate(new_node)
    except ValidationError:
        print("[ERROR] Node baru tidak valid untuk Edge Generation.")
        return {"edges": final_relations_to_insert, "token_usage": default_token_usage}
    
    conn_save = None
    total_tokens = default_token_usage.copy()

    try:
        conn_save = await _get_pg_connection()

        for existing_art_node in existing_article_nodes:
            existing_art_id = existing_art_node['id']
            existing_art_title = existing_art_node['title']
            
            existing_art_details = {
                "background": existing_art_node.get('att_background', ''),
                "methodology": existing_art_node.get('att_method', ''),
                "purpose": existing_art_node.get('att_goal', ''),
                "future_research": existing_art_node.get('att_future', ''),
                "research_gap": existing_art_node.get('att_gaps', '')
            }

            aspects_to_check = {
                "background": "SERUPA_LATAR_BELAKANG",
                "methodology": "SERUPA_METODOLOGI",
                "purpose": "SERUPA_TUJUAN",
                "future_research": "SERUPA_PENELITIAN_LANJUT",
                "research_gap": "SERUPA_GAP_PENELITIAN",
            }

            for aspect_key, relation_type in aspects_to_check.items():
                current_article_details = {
                    "background": main_article_node.att_background,
                    "methodology": main_article_node.att_method,
                    "purpose": main_article_node.att_goal,
                    "future_research": main_article_node.att_future,
                    "research_gap": main_article_node.att_gaps
                }

                if not current_article_details.get(aspect_key, '').strip() or not existing_art_details.get(aspect_key, '').strip():
                    continue

                try:
                    verification_input = {
                        "article1_id": main_article_node.id,
                        "article1_details": current_article_details,
                        "article2_id": existing_art_id,
                        "article2_details": existing_art_details,
                        "format_instruction": FORMAT_INSTRUCTION_REL_VERIFY
                    }

                    relation_chain_to_use = RELATION_CHAINS[relation_type]
                    
                    verification_output = await asyncio.wait_for(relation_chain_to_use.ainvoke(verification_input), timeout=30)
                    verified_rel_str = verification_output.content.strip().lstrip('```json').rstrip('```').strip()
                    
                    # Track Token Usage
                    if hasattr(verification_output, 'usage_metadata'):
                        usage = verification_output.usage_metadata
                        total_tokens['input_tokens'] += usage.get('input_tokens', 0)
                        total_tokens['output_tokens'] += usage.get('output_tokens', 0)
                        total_tokens['total_tokens'] += usage.get('total_tokens', 0)

                    if not verified_rel_str or not verified_rel_str.strip():
                        continue
                        
                    verified_rel_dict = json.loads(verified_rel_str)
                    verified_type = verified_rel_dict.get("type", "").strip()

                    if verified_type == relation_type:
                        label = relation_type.replace('_', ' ').title()
                        
                        # --- PERUBAHAN UTAMA: Hapus 'color' dari INSERT Query ---
                        await conn_save.execute("""
                            INSERT INTO edges_kg (id, "fromId", "toId", relation, label, context)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (id) DO NOTHING;
                        """, 
                            str(uuid.uuid4()), # $1: ID unik
                            main_article_node.id, # $2: fromId
                            existing_art_id, # $3: toId
                            relation_type, # $4: relation
                            label, # $5: label
                            verified_rel_dict.get("context", f"Ditemukan kemiripan pada {aspect_key} antara '{main_article_node.title}' dan '{existing_art_title}'.") # $6: context
                        )
                        print(f"    [SAVED] Edge '{main_article_node.id}' -[{relation_type}]-> '{existing_art_id}' berhasil disimpan.")
                        
                        final_relations_to_insert.append({
                            "from": main_article_node.id,
                            "to": existing_art_id,
                            "relation": relation_type,
                            "label": label
                        })

                except asyncio.TimeoutError:
                    print(f"  [kg_processor] Timeout untuk verifikasi {relation_type} antara '{main_article_node.id}' dan '{existing_art_id}'.")
                    continue
                except json.JSONDecodeError as llm_parse_e:
                    print(f"  [kg_processor] Error parsing LLM output untuk {relation_type}: {llm_parse_e}.")
                    continue
                except Exception as llm_e:
                    print(f"  [kg_processor] Error memanggil LLM untuk verifikasi {relation_type}: {llm_e}")
                    continue

        return {
            "edges": final_relations_to_insert,
            "token_usage": total_tokens
        }

    except Exception as e:
        print(f"[ERROR] Gagal memproses edges: {e}")
        return {
            "edges": [],
            "token_usage": default_token_usage,
            "error": str(e)
        }
    finally:
        if conn_save:
            await conn_save.close()

# @app.post("/api/suggestions")
async def get_suggestions(request: SuggestionRequest):
    try:
        context = request.context or {}
        node_ids = context.get("nodeIds", [])
        edge_ids = context.get("edgeIds", [])

        # 1. Fetch node metadata from your graph context endpoint
        context_data = {}
        if node_ids:
            context_data = await graph_service.get_graph_context(node_ids=node_ids, edge_ids=edge_ids)
        
        # 2. Extract node titles or labels
        node_titles = [n.get("title") or n.get("label") for n in context_data.get("nodes", []) if n.get("title") or n.get("label")]
        # 1. Ambil judul dari node kalau ada
        if node_titles:
            topic_description = ", ".join(node_titles[:3])

        # 2. Kalau tidak ada node, ambil query user kalau cukup informatif
        elif request.query and len(request.query.strip()) > 5 and request.query.lower() not in ["general", "umum", "topik umum"]:
            topic_description = "topik riset ilmiah atau brainstorming"

        # 3. Fallback terakhir: topik default
        else:
            topic_description = random.choice([
                "potensi riset interdisipliner di era digital",
                "inovasi dalam metode penelitian akademik",
                "tren terbaru dalam kecerdasan buatan",
                "tantangan etis dalam publikasi ilmiah",
                "kolaborasi riset global di masa depan"
            ])


        # 3. Buat prompt kontekstual
        prompt = f"""Kamu adalah asisten brainstorming. Buat 5 saran eksploratif berdasarkan topik: "{topic_description}"

Saran bisa berupa:
- pertanyaan kritis
- perbandingan ide
- gap yang belum dijawab
- metode alternatif
- contoh penerapan di bidang lain

Format:
1. ...
2. ...
3. ...
        """

        print(f"🧠 Prompt yang digunakan:\n{prompt[:200]}...")

        # 4. Generate suggestion
        suggestions = await system["action"].generate_suggestions(prompt)

        clean_suggestions = []
        for s in suggestions:
            if s and len(s.strip()) > 5:
                if s[0].isdigit() and s[1:3] == '. ':
                    clean_suggestions.append(s[3:].strip())
                else:
                    clean_suggestions.append(s.strip())

        return {
            "success": True,
            "suggestions": clean_suggestions[:5] or [
                f"Apa gap dari {topic_description}?",
                f"Studi kasus {topic_description}",
                f"Pendekatan alternatif untuk {topic_description}",
                f"Permasalahan utama dalam {topic_description}",
                f"Aplikasi nyata dari {topic_description}"
            ]
        }

    except Exception as e:
        print(f"❌ Error while generating suggestions: {e}")
        return {"success": False, "suggestions": []}

    
# @app.post("/api/suggestions/followup")
async def get_followup_suggestions(request: FollowupRequest):
    try:
        # Buat context dari conversation history
        conversation_context = ""
        if request.conversationHistory:
            recent_messages = request.conversationHistory[-3:]  # Ambil 3 terakhir
            conversation_context = "\n".join([
                f"{'User' if msg.get('sender') == 'user' else 'AI'}: {msg.get('text', '')[:100]}..."
                for msg in recent_messages
            ])
        
        # Buat context dari nodes/edges
        node_context = ""
        if request.context and (request.context.get("nodeIds") or request.context.get("edgeIds")):
            node_context = f"\nContext: Berdasarkan {len(request.context.get('nodeIds', []))} nodes dan {len(request.context.get('edgeIds', []))} edges"
        
        prompt = f"""Berdasarkan jawaban AI terakhir, buat 5 pertanyaan lanjutan yang relevan dan mendalam:

Jawaban AI terakhir: "{request.lastMessage[:300]}..."

Konteks percakapan sebelumnya:
{conversation_context}
{node_context}

Buat pertanyaan follow-up yang:
- Menggali lebih dalam dari jawaban yang diberikan
- Mengeksplorasi aspek praktis atau implementasi
- Menanyakan contoh konkret atau studi kasus
- Mempertanyakan hubungan dengan konsep lain
- Menanyakan tentang tantangan atau limitasi

Format: berikan 5 pertanyaan dengan format numerik (1. ... 2. ...)
Setiap pertanyaan maksimal 10 kata dan berupa kalimat tanya."""
        
        suggestions = await system["action"].generate_followup_suggestions(prompt)
        
        # Filter dan bersihkan suggestions
        clean_suggestions = []
        for s in suggestions:
            if len(s) > 10:
                # Hapus prefix nomor jika ada
                if s[0].isdigit() and len(s) > 3 and s[1:3] == '. ':
                    clean_suggestions.append(s[3:].strip())
                else:
                    clean_suggestions.append(s.strip())
        
        return {
            "success": True,
            "suggestions": clean_suggestions[:3] or [
                f"Bagaimana cara mengimplementasikan konsep ini?",
                f"Apa tantangan utama dalam penerapannya?",
                f"Bisakah berikan contoh kasus nyata?",
                f"Bagaimana perkembangan terbaru di bidang ini?",
                f"Apa perbedaannya dengan pendekatan lain?"
            ]
        }
        
    except Exception as e:
        print(f"❌ Followup Error: {str(e)}")
        return {
            "success": False,
            "suggestions": [
                "Bisakah dijelaskan lebih detail?",
                "Bagaimana cara praktis menerapkannya?",
                "Apa contoh kasus penggunaannya?",
                "Bagaimana tren perkembangannya?",
                "Apa kelebihan dan kekurangannya?"
            ]
        }

# --- Endpoint Tambahan ---
@app.post("/api/feedback")
async def submit_feedback(interaction_id: str, feedback: str, rating: int):
    """Catat feedback pengguna"""
    try:
        system["learning"].add_user_feedback(interaction_id, feedback, rating)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system-info")
async def system_info():
    """Get info tentang sistem"""
    return {
        "providers": list(system["mcp"].providers.keys()),
        "learning_metrics": system["learning"].get_learning_insights()
    }

@app.get("/api/debug/chroma-metadata")
async def debug_chroma_metadata(limit: int = 3):
    """Enhanced debug endpoint with proper error handling"""
    try:
        # 1. Get ChromaDB collection
        collection = system["chroma_db"].client.get_collection(COLLECTION_NAME)
        
        # 2. Get records with proper type handling
        # records = collection.get(
        #     limit=min(limit, 10),
        #     include=["metadatas", "documents"]
        # )

        if limit == -1:
            records = collection.get(include=["metadatas", "documents"])
        else:
            records = collection.get(limit=limit, include=["metadatas", "documents"])

        total_chars = sum(len(doc) for doc in records["documents"] if doc)

        # 3. Safely format samples
        samples = []
        for i in range(min(len(records["ids"]), min(limit, 10))):
            doc = records["documents"][i] if i < len(records["documents"]) else None
            meta = records["metadatas"][i] if i < len(records["metadatas"]) else {}
            
            samples.append({
                "id": records["ids"][i],
                "metadata": meta,
                "document_preview": f"{doc[:100]}..." if doc else None
            })

        return {
            "success": True,
            "collection": COLLECTION_NAME,
            "count": len(records["ids"]),
            "total_characters": total_chars,
            "samples": samples
        }

    except Exception as e:
        logger.error(f"Debug error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to retrieve metadata: {str(e)}",
            "details": {
                "collection": COLLECTION_NAME,
                "available_collections": system["chroma_db"].client.list_collections()
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)