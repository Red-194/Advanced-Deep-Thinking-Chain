import asyncio
import torch, gc
import concurrent.futures

# LangChain imports
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableParallel

from models import ThoughtStage, ThoughtProcess

# HuggingFace imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    TextStreamer
)

class Config:
    """Configuration management"""
    DEFAULTS = {
        "model_name": "microsoft/phi-2",
        "temperature": 0.7,
        "max_new_tokens": 1024,
        "use_gpu": True,
        "quantize": True,
        "streaming": True,
        "max_concurrent": 2
    }
    
    def __init__(self, **kwargs):
        self.config = self.DEFAULTS.copy()
        self.config.update(kwargs)
    
    def __getattr__(self, name):
        return self.config.get(name)

class AdvancedDeepThinkingChain:
    """Enhanced deep thinking system with parallel processing"""

    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        self.device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Initializing with {self.config.model_name} on {self.device}...")

        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.llm = None
        self.streaming_llm = None

        self._load_model()
        self._create_llm()
        self._create_chains()

        print("System initialized successfully!")

    def _load_model(self):
        """Load model with optimization for GPU/CPU and optional quantization"""

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model_kwargs = {"trust_remote_code": True}

        if self.config.quantize and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs)

        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=TextStreamer(self.tokenizer, skip_prompt=True) if self.config.streaming else None
        )

    def _create_llm(self):
        """Create LLM instances"""
        llm_kwargs = {
            "pipeline": self.pipe,
            "model_kwargs": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_new_tokens
            }
        }

        self.llm = HuggingFacePipeline(**llm_kwargs)

        if self.config.streaming:
            llm_kwargs["callbacks"] = [StreamingStdOutCallbackHandler()]
            self.streaming_llm = HuggingFacePipeline(**llm_kwargs)

    def _create_chains(self):
        """Create thinking chains with parallel capabilities"""
        # Analysis Chain
        analysis_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Analyze systematically:\n\nQuestion: {question}\n\nProvide:\n1. Key concepts\n2. Question type\n3. Assumptions\n4. Sub-questions\n\nAnalysis:"""
        )
        self.analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt, output_key="analysis")

        # Research Chain
        research_prompt = PromptTemplate(
            input_variables=["question", "analysis"],
            template="""Research perspectives:\n\nQuestion: {question}\nAnalysis: {analysis}\n\nExplore:\n1. Viewpoints\n2. History\n3. Debates\n4. Applications\n5. Implications\n\nResearch:"""
        )
        self.research_chain = LLMChain(llm=self.llm, prompt=research_prompt, output_key="research")

        # Critique Chain
        critique_prompt = PromptTemplate(
            input_variables=["question", "analysis", "research"],
            template="""Critically evaluate:\n\nQuestion: {question}\nAnalysis: {analysis}\nResearch: {research}\n\nExamine:\n1. Assumptions\n2. Fallacies\n3. Evidence\n4. Limitations\n5. Alternatives\n\nCritique:"""
        )
        self.critique_chain = LLMChain(llm=self.llm, prompt=critique_prompt, output_key="critique")

        # Creative Chain
        creative_prompt = PromptTemplate(
            input_variables=["question", "analysis", "research"],
            template="""Generate insights:\n\nQuestion: {question}\nAnalysis: {analysis}\nResearch: {research}\n\nProvide:\n1. Connections\n2. Metaphors\n3. Solutions\n4. Experiments\n5. Perspectives\n\nCreative:"""
        )
        self.creative_chain = LLMChain(llm=self.llm, prompt=creative_prompt, output_key="creative")

        # Synthesis Chain
        synthesis_prompt = PromptTemplate(
            input_variables=["question", "analysis", "research", "critique", "creative"],
            template="""Synthesize:\n\nQuestion: {question}\nAnalysis: {analysis}\nResearch: {research}\nCritique: {critique}\nCreative: {creative}\n\nAnswer:\n1. Addresses question\n2. Integrates insights\n3. Acknowledges nuance\n4. Gives examples\n5. Offers implications\n\nSynthesis:"""
        )
        self.synthesis_chain = LLMChain(
            llm=self.streaming_llm if self.config.streaming else self.llm,
            prompt=synthesis_prompt,
            output_key="synthesis"
        )

        # Parallel execution for independent chains
        self.parallel_chains = RunnableParallel(
            critique=self.critique_chain,
            creative=self.creative_chain
        )

        # Conversation chain
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=ConversationBufferMemory(),
            verbose=False
        )

    async def _run_chain_async(self, chain, inputs):
        """Run a chain asynchronously"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, lambda: chain.run(inputs))

    async def think_deeply(self, question: str) -> ThoughtProcess:
        """Process question with parallel execution where possible"""
        print("\n🧠 Starting Deep Thinking...\n")

        stages = []

        # Sequential steps
        analysis = await self._run_chain_async(self.analysis_chain, {"question": question})
        stages.append(ThoughtStage(name="analysis", content=analysis))

        research = await self._run_chain_async(self.research_chain, {
            "question": question,
            "analysis": analysis
        })
        stages.append(ThoughtStage(name="research", content=research))

        # Parallel steps for critique and creative
        critique_task = self._run_chain_async(self.critique_chain, {
            "question": question,
            "analysis": analysis,
            "research": research
        })
        creative_task = self._run_chain_async(self.creative_chain, {
            "question": question,
            "analysis": analysis,
            "research": research
        })
        critique, creative = await asyncio.gather(critique_task, creative_task)
        stages.extend([ThoughtStage(name="critique", content=critique), ThoughtStage(name="creative", content=creative)])
        synthesis = await self._run_chain_async(self.synthesis_chain, {
            "question": question,
            "analysis": analysis,
            "research": research,
            "critique": critique,
            "creative": creative
        })
        stages.append(ThoughtStage(name="synthesis", content=synthesis))

        return ThoughtProcess(stages=stages)

    async def quick_think(self, question: str) -> str:
        """Quick response mode"""
        prompt = f"Answer thoughtfully:\nQuestion:{question}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.llm(prompt))

    async def follow_up(self, question: str) -> str:
        """Follow-up with conversation context"""
        return await self._run_chain_async(self.conversation_chain, {"input": question})

    def unload_model(self):
        """Aggressively unload model and clear memory"""
        for attr in ["model", "pipe", "llm", "streaming_llm", "tokenizer"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)

        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("Model and resources unloaded successfully.")
