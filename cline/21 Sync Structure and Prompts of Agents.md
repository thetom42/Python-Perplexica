**User:**

<task>
Once more compare the Python agents with the Typescript ones and make sure that the implementations are in sync. Especially make sure that the prompts that are being used in the Typescript agents are in the Python version as well.
</task>
<environment_details>
# VSCode Visible Files
backend/agents/academic_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py

# Current Working Directory (/Users/the/src/repos/thetom42/github/python-perplexica) Files
.dockerignore
.flake8
.gitignore
.mypy.ini
.prettierignore
.prettierrc.js
.pylintrc
app.dockerfile
backend.dockerfile
CONTRIBUTING.md
docker-compose.yaml
drizzle.config.ts
LICENSE
package.json
README.md
README.original.md
sample.config.toml
ts-backend.dockerfile
tsconfig.json
yarn.lock
backend/
backend/.gitignore
backend/alembic.ini
backend/config.py
backend/conftest.py
backend/locustfile.py
backend/main.py
backend/pytest.ini
backend/README.md
backend/requirements.txt
backend/setup.py
backend/agents/
backend/agents/__init__.py
backend/agents/abstract_agent.py
backend/agents/academic_search_agent.py
backend/agents/image_search_agent.py
backend/agents/reddit_search_agent.py
backend/agents/suggestion_generator_agent.py
backend/agents/video_search_agent.py
backend/agents/web_search_agent.py
backend/agents/wolfram_alpha_search_agent.py
backend/agents/writing_assistant_agent.py
backend/agents/youtube_search_agent.py
backend/alembic/
backend/alembic/env.py
backend/alembic/versions/
backend/db/
backend/db/crud.py
backend/db/database.py
backend/db/models.py
backend/lib/
backend/lib/huggingface_transformer.py
backend/lib/link_document.py
backend/lib/searxng.py
backend/lib/output_parsers/
backend/lib/providers/
backend/locust-stubs/
backend/locust-stubs/__init__.pyi
backend/logs/
backend/routes/
backend/routes/__init__.py
backend/routes/chats.py
backend/routes/config.py
backend/routes/images.py
backend/routes/models.py
backend/routes/search.py
backend/routes/suggestions.py
backend/routes/videos.py
backend/tests/
backend/tests/__init__.py
backend/tests/test_academic_search_agent.py
backend/tests/test_compute_similarity.py
backend/tests/test_db_crud.py
backend/tests/test_image_search_agent.py
backend/tests/test_integration.py
backend/tests/test_reddit_search_agent.py
backend/tests/test_search_route.py
backend/tests/test_suggestion_generator_agent.py
backend/tests/test_video_search_agent.py
backend/tests/test_web_search_agent.py
backend/tests/test_wolfram_alpha_search_agent.py
backend/tests/test_writing_assistant_agent.py
backend/tests/test_youtube_search_agent.py
backend/utils/
backend/utils/cache.py
backend/utils/compute_similarity.py
backend/utils/format_history.py
backend/utils/logger.py
backend/websocket/
backend/websocket/connection_manager.py
backend/websocket/message_handler.py
backend/websocket/websocket_server.py
cline/
cline/01 Explain Structure of Typescript Backend.md
cline/02 Transform Backend to Python.md
cline/03 Transfer All Chat Models.md
cline/04 Improve Python Code and Add Unittests.md
cline/05 Add Devcontainer.md
cline/06 Better Sync of Route Interface with the Typescript Version.md
cline/07 Sync Agents Part 1.md
cline/08 Sync Agents Part 2.md
cline/09 Sync Agents Part 3.md
cline/10 Suggesting Refactorings for Agents.md
cline/11 Adapt Unittests for Agents.md
cline/12 ABC for Agents.md
cline/13 Complete Code Review.md
cline/14 Add Launch Configurations.md
cline/15 Fix Linting Problems in Agents.md
cline/16 Fix Linting Problems in Alembic.md
cline/17 Fix Linting Problems in lib.md
cline/18 Fix Linting Problems in routes.md
cline/19 Fix Linting Problems in test.md
cline/20 Create Pytest Configuration.md
data/
data/.gitignore
docs/
docs/API/
docs/API/SEARCH.md
docs/architecture/
docs/architecture/README.md
docs/architecture/WORKING.md
docs/installation/
docs/installation/NETWORKING.md
docs/installation/UPDATING.md
logs/
src/
src/app.ts
src/config.ts
src/agents/
src/agents/academicSearchAgent.ts
src/agents/imageSearchAgent.ts
src/agents/redditSearchAgent.ts
src/agents/suggestionGeneratorAgent.ts
src/agents/videoSearchAgent.ts
src/agents/webSearchAgent.ts
src/agents/wolframAlphaSearchAgent.ts
src/agents/writingAssistant.ts
src/agents/youtubeSearchAgent.ts
src/db/
src/db/index.ts
src/db/schema.ts
src/lib/
src/lib/huggingfaceTransformer.ts
src/lib/linkDocument.ts
src/lib/searxng.ts
src/lib/outputParsers/
src/lib/providers/
src/routes/
src/routes/chats.ts
src/routes/config.ts
src/routes/images.ts
src/routes/index.ts
src/routes/models.ts
src/routes/search.ts
src/routes/suggestions.ts
src/routes/videos.ts
src/utils/
src/utils/computeSimilarity.ts
src/utils/formatHistory.ts
src/utils/logger.ts
src/websocket/
src/websocket/connectionManager.ts
src/websocket/index.ts
src/websocket/messageHandler.ts
src/websocket/websocketServer.ts
ui/
ui/.env.example
ui/.eslintrc.json
ui/.gitignore
ui/.prettierrc.js
ui/next.config.mjs
ui/package.json
ui/postcss.config.js
ui/tailwind.config.ts
ui/tsconfig.json
ui/yarn.lock
ui/app/
ui/app/favicon.ico
ui/app/globals.css
ui/app/layout.tsx
ui/app/page.tsx
ui/app/c/
ui/app/library/
ui/components/
ui/components/Chat.tsx
ui/components/ChatWindow.tsx
ui/components/DeleteChat.tsx
ui/components/EmptyChat.tsx
ui/components/EmptyChatMessageInput.tsx
ui/components/Layout.tsx
ui/components/MessageBox.tsx
ui/components/MessageBoxLoading.tsx
ui/components/MessageInput.tsx
ui/components/MessageSources.tsx
ui/components/Navbar.tsx
ui/components/SearchImages.tsx
ui/components/MessageActions/
ui/components/MessageInputActions/
ui/lib/
ui/public/

(File list truncated. Use list_files on specific subdirectories if you need to explore further.)
</environment_details>

---

**Assistant:**

<thinking>
Let me analyze this task systematically:
1. I need to compare Python agents with TypeScript agents to ensure sync
2. Special focus on prompts used in agents
3. I should first get a list of all agents in both languages
4. Then I can compare them one by one

Let me start by listing the code definitions in both agent directories to get an overview.
</thinking>

<list_code_definition_names>
<path>src/agents</path>
</list_code_definition_names>

---

**User:**

[list_code_definition_names for 'src/agents'] Result:
No source code definitions found.
<environment_details>
# VSCode Visible Files
backend/agents/academic_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
I apologize - I need to read the actual agent files to compare them. Let me start with reading one Python agent and its TypeScript counterpart to compare them directly.

Let me start with the academic search agent in both languages.
</thinking>

<read_file>
<path>backend/agents/academic_search_agent.py</path>
</read_file>

---

**User:**

[read_file for 'backend/agents/academic_search_agent.py'] Result:
"""
Academic Search Agent Module

This module provides functionality for performing academic searches using multiple academic search
engines (arXiv, Google Scholar, PubMed) and processing the results using a language model to
generate concise, well-cited summaries.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

BASIC_ACADEMIC_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: How does stable diffusion work?
Rephrased: Stable diffusion working

2. Follow up question: What is linear algebra?
Rephrased: Linear algebra

3. Follow up question: What is the third law of thermodynamics?
Rephrased: Third law of thermodynamics

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class AcademicSearchAgent(AbstractAgent):
    """Agent for performing academic searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the academic search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_ACADEMIC_SEARCH_RETRIEVER_PROMPT)

        retriever_chain = prompt | self.llm

        async def process_llm_output(rephrased_query: str) -> Dict[str, Any]:
            """Process the LLM output and perform the academic search."""
            if rephrased_query.strip() == "not_needed":
                return {"query": "", "docs": []}

            try:
                search_results = await search_searxng(rephrased_query, {
                    "language": "en",
                    "engines": ["arxiv", "google scholar", "pubmed"]
                })

                documents = [
                    Document(
                        page_content=result["content"],
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            "img_src": result.get("img_src")
                        }
                    )
                    for result in search_results["results"]
                ]

                return {"query": rephrased_query, "docs": documents}
            except Exception as e:
                logger.error("Error in academic search: %s", str(e))
                return {"query": rephrased_query, "docs": []}

        chain = RunnableSequence([
            {
                "rephrased_query": retriever_chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_llm_output(x["rephrased_query"])
        ])

        return chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the academic search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def process_retriever_output(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the retriever output and generate the final response."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "response": "I'm not sure how to help with that. Could you please ask a specific question?",
                        "sources": []
                    }

                reranked_docs = await self.rerank_docs(
                    retriever_result["query"],
                    retriever_result["docs"]
                )

                context = await self.process_docs(reranked_docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(retriever_result["query"], context)),
                    ("human", "{query}"),
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({
                    "query": retriever_result["query"],
                    "context": context
                })

                return {
                    "response": response.content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in reranked_docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "response": "An error occurred while processing the academic search results",
                    "sources": []
                }

        return RunnableSequence([process_retriever_output])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Academic', this means you will be searching for academic papers and articles on the web.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from a search engine and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_academic_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle an academic search query and generate streaming responses.

    Args:
        query: The academic search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = AcademicSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result

<environment_details>
# VSCode Visible Files
backend/agents/academic_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Now let me read the TypeScript version of the academic search agent to compare them.
</thinking>

<read_file>
<path>src/agents/academicSearchAgent.ts</path>
</read_file>

---

**User:**

[read_file for 'src/agents/academicSearchAgent.ts'] Result:
import { BaseMessage } from '@langchain/core/messages';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import { searchSearxng } from '../lib/searxng';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import computeSimilarity from '../utils/computeSimilarity';
import logger from '../utils/logger';
import { IterableReadableStream } from '@langchain/core/utils/stream';

const basicAcademicSearchRetrieverPrompt = `
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return \`not_needed\` as the response.

Example:
1. Follow up question: How does stable diffusion work?
Rephrased: Stable diffusion working

2. Follow up question: What is linear algebra?
Rephrased: Linear algebra

3. Follow up question: What is the third law of thermodynamics?
Rephrased: Third law of thermodynamics

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
`;

const basicAcademicSearchResponsePrompt = `
    You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Academic', this means you will be searching for academic papers and articles on the web.

    Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
    You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
    You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
    Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
    You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
    Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
    However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

    Anything inside the following \`context\` HTML block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to 
    talk about the context in your response. 

    <context>
    {context}
    </context>

    If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
    Anything between the \`context\` is retrieved from a search engine and is not a part of the conversation with the user. Today's date is ${new Date().toISOString()}
`;

const strParser = new StringOutputParser();

const handleStream = async (
  stream: IterableReadableStream<StreamEvent>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalSourceRetriever'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'sources', data: event.data.output }),
      );
    }
    if (
      event.event === 'on_chain_stream' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

type BasicChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const createBasicAcademicSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicAcademicSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'en',
        engines: [
          'arxiv',
          'google scholar',
          'pubmed',
        ],
      });

      const documents = res.results.map(
        (result) =>
          new Document({
            pageContent: result.content,
            metadata: {
              title: result.title,
              url: result.url,
              ...(result.img_src && { img_src: result.img_src }),
            },
          }),
      );

      return { query: input, docs: documents };
    }),
  ]);
};

const createBasicAcademicSearchAnsweringChain = (
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const basicAcademicSearchRetrieverChain =
    createBasicAcademicSearchRetrieverChain(llm);

  const processDocs = async (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
  };

  const rerankDocs = async ({
    query,
    docs,
  }: {
    query: string;
    docs: Document[];
  }) => {
    if (docs.length === 0) {
      return docs;
    }

    const docsWithContent = docs.filter(
      (doc) => doc.pageContent && doc.pageContent.length > 0,
    );

    if (optimizationMode === 'speed') {
      return docsWithContent.slice(0, 15);
    } else if (optimizationMode === 'balanced') {
      console.log('Balanced mode');
      const [docEmbeddings, queryEmbedding] = await Promise.all([
        embeddings.embedDocuments(
          docsWithContent.map((doc) => doc.pageContent),
        ),
        embeddings.embedQuery(query),
      ]);

      const similarity = docEmbeddings.map((docEmbedding, i) => {
        const sim = computeSimilarity(queryEmbedding, docEmbedding);

        return {
          index: i,
          similarity: sim,
        };
      });

      const sortedDocs = similarity
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 15)
        .map((sim) => docsWithContent[sim.index]);

      return sortedDocs;
    }
  };

  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: BasicChainInput) => input.query,
      chat_history: (input: BasicChainInput) => input.chat_history,
      context: RunnableSequence.from([
        (input) => ({
          query: input.query,
          chat_history: formatChatHistoryAsString(input.chat_history),
        }),
        basicAcademicSearchRetrieverChain
          .pipe(rerankDocs)
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicAcademicSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicAcademicSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = new eventEmitter();

  try {
    const basicAcademicSearchAnsweringChain =
      createBasicAcademicSearchAnsweringChain(
        llm,
        embeddings,
        optimizationMode,
      );

    const stream = basicAcademicSearchAnsweringChain.streamEvents(
      {
        chat_history: history,
        query: query,
      },
      {
        version: 'v1',
      },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
      'error',
      JSON.stringify({ data: 'An error has occurred please try again later' }),
    );
    logger.error(`Error in academic search: ${err}`);
  }

  return emitter;
};

const handleAcademicSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = basicAcademicSearch(
    message,
    history,
    llm,
    embeddings,
    optimizationMode,
  );
  return emitter;
};

export default handleAcademicSearch;

<environment_details>
# VSCode Visible Files
backend/agents/academic_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
After comparing both implementations, I notice several key points:

1. The prompts are mostly in sync, but there are some differences:
   - The basic retriever prompt (BASIC_ACADEMIC_SEARCH_RETRIEVER_PROMPT) is identical in both versions
   - The response prompt has some differences in the TypeScript version that should be added to Python:
     - Additional clarification about citation numbers
     - More explicit instruction about context handling

2. The core functionality is similar but structured differently:
   - Python uses a class-based approach (AcademicSearchAgent)
   - TypeScript uses functional composition

Let me update the Python version to ensure the prompts are fully in sync.
</thinking>

<write_to_file>
<path>backend/agents/academic_search_agent.py</path>
<content>"""
Academic Search Agent Module

This module provides functionality for performing academic searches using multiple academic search
engines (arXiv, Google Scholar, PubMed) and processing the results using a language model to
generate concise, well-cited summaries.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

BASIC_ACADEMIC_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: How does stable diffusion work?
Rephrased: Stable diffusion working

2. Follow up question: What is linear algebra?
Rephrased: Linear algebra

3. Follow up question: What is the third law of thermodynamics?
Rephrased: Third law of thermodynamics

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class AcademicSearchAgent(AbstractAgent):
    """Agent for performing academic searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the academic search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_ACADEMIC_SEARCH_RETRIEVER_PROMPT)

        retriever_chain = prompt | self.llm

        async def process_llm_output(rephrased_query: str) -> Dict[str, Any]:
            """Process the LLM output and perform the academic search."""
            if rephrased_query.strip() == "not_needed":
                return {"query": "", "docs": []}

            try:
                search_results = await search_searxng(rephrased_query, {
                    "language": "en",
                    "engines": ["arxiv", "google scholar", "pubmed"]
                })

                documents = [
                    Document(
                        page_content=result["content"],
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            "img_src": result.get("img_src")
                        }
                    )
                    for result in search_results["results"]
                ]

                return {"query": rephrased_query, "docs": documents}
            except Exception as e:
                logger.error("Error in academic search: %s", str(e))
                return {"query": rephrased_query, "docs": []}

        chain = RunnableSequence([
            {
                "rephrased_query": retriever_chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_llm_output(x["rephrased_query"])
        ])

        return chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the academic search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def process_retriever_output(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the retriever output and generate the final response."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "response": "I'm not sure how to help with that. Could you please ask a specific question?",
                        "sources": []
                    }

                reranked_docs = await self.rerank_docs(
                    retriever_result["query"],
                    retriever_result["docs"]
                )

                context = await self.process_docs(reranked_docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(retriever_result["query"], context)),
                    ("human", "{query}"),
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({
                    "query": retriever_result["query"],
                    "context": context
                })

                return {
                    "response": response.content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in reranked_docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "response": "An error occurred while processing the academic search results",
                    "sources": []
                }

        return RunnableSequence([process_retriever_output])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Academic', this means you will be searching for academic papers and articles on the web.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
        However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

        Anything inside the following `context` HTML block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to 
        talk about the context in your response. 

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from a search engine and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_academic_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle an academic search query and generate streaming responses.

    Args:
        query: The academic search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = AcademicSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
</content>
</write_to_file>

---

**User:**

[write_to_file for 'backend/agents/academic_search_agent.py'] Result:
The content was successfully saved to backend/agents/academic_search_agent.py.
<environment_details>
# VSCode Visible Files
backend/agents/academic_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me check another agent pair to ensure they are in sync. I'll look at the web search agent next since it's another commonly used one.
</thinking>

<read_file>
<path>backend/agents/web_search_agent.py</path>
</read_file>

---

**User:**

[read_file for 'backend/agents/web_search_agent.py'] Result:
"""
Web Search Agent Module

This module provides functionality for performing web searches, rephrasing queries,
and generating responses based on search results or webpage content. It includes
specialized handling for webpage summarization and link-based content retrieval.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from lib.searxng import search_searxng
from lib.link_document import get_documents_from_links
from utils.logger import logger
from lib.output_parsers.line_output_parser import LineOutputParser
from lib.output_parsers.list_line_output_parser import ListLineOutputParser
from agents.abstract_agent import AbstractAgent

BASIC_SEARCH_RETRIEVER_PROMPT = """
You are an AI question rephraser. You will be given a conversation and a follow-up question,  you will have to rephrase the follow up question so it is a standalone question and can be used by another LLM to search the web for information to answer it.
If it is a smple writing task or a greeting (unless the greeting contains a question after it) like Hi, Hello, How are you, etc. than a question then you need to return `not_needed` as the response (This is because the LLM won't need to search the web for finding information on this topic).
If the user asks some question from some URL or wants you to summarize a PDF or a webpage (via URL) you need to return the links inside the `links` XML block and the question inside the `question` XML block. If the user wants to you to summarize the webpage or the PDF you need to return `summarize` inside the `question` XML block in place of a question and the link to summarize in the `links` XML block.

<conversation>
{chat_history}
</conversation>

Follow up question: {query}
Rephrased question:
"""

class WebSearchAgent(AbstractAgent):
    """Agent for performing web searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the web search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_SEARCH_RETRIEVER_PROMPT)

        chain = prompt | self.llm

        async def process_output(llm_output: Any) -> Dict[str, Any]:
            """Process the LLM output and perform the web search."""
            try:
                output_text = llm_output.content if hasattr(llm_output, 'content') else str(llm_output)
                links_parser = ListLineOutputParser(key="links")
                question_parser = LineOutputParser(key="question")

                links = await links_parser.parse(output_text)
                question = await question_parser.parse(output_text)

                if question == "not_needed":
                    return {"query": "", "docs": []}

                if links:
                    if not question:
                        question = "summarize"

                    docs = []
                    link_docs = await get_documents_from_links({"links": links})
                    doc_groups = []

                    for doc in link_docs:
                        url_doc = next(
                            (d for d in doc_groups if d.metadata["url"] == doc.metadata["url"] and d.metadata.get("totalDocs", 0) < 10),
                            None
                        )

                        if not url_doc:
                            doc.metadata["totalDocs"] = 1
                            doc_groups.append(doc)
                        else:
                            url_doc.page_content += f"\n\n{doc.page_content}"
                            url_doc.metadata["totalDocs"] = url_doc.metadata.get("totalDocs", 0) + 1

                    for doc in doc_groups:
                        try:
                            summary_prompt = ChatPromptTemplate.from_messages([
                                ("system", """
                                You are a web search summarizer, tasked with summarizing a piece of text retrieved from a web search. Your job is to summarize the 
                                text into a detailed, 2-4 paragraph explanation that captures the main ideas and provides a comprehensive answer to the query.
                                If the query is "summarize", you should provide a detailed summary of the text. If the query is a specific question, you should answer it in the summary.

                                - **Journalistic tone**: The summary should sound professional and journalistic, not too casual or vague.
                                - **Thorough and detailed**: Ensure that every key point from the text is captured and that the summary directly answers the query.
                                - **Not too lengthy, but detailed**: The summary should be informative but not excessively long. Focus on providing detailed information in a concise format.
                                """),
                                ("human", """
                                <query>
                                {query}
                                </query>

                                <text>
                                {text}
                                </text>

                                Make sure to answer the query in the summary.
                                """)
                            ])

                            summary_chain = summary_prompt | self.llm

                            summary = await summary_chain.invoke({
                                "query": question,
                                "text": doc.page_content
                            })

                            docs.append(Document(
                                page_content=summary.content,
                                metadata={
                                    "title": doc.metadata["title"],
                                    "url": doc.metadata["url"]
                                }
                            ))
                        except Exception as e:
                            logger.error("Error summarizing document: %s", str(e))
                            docs.append(doc)

                    return {"query": question, "docs": docs}
                else:
                    res = await search_searxng(question, {
                        "language": "en"
                    })

                    documents = [
                        Document(
                            page_content=result["content"],
                            metadata={
                                "title": result["title"],
                                "url": result["url"],
                                **({"img_src": result["img_src"]} if "img_src" in result else {})
                            }
                        )
                        for result in res["results"]
                    ]

                    return {"query": question, "docs": documents}
            except Exception as e:
                logger.error("Error in web search/processing: %s", str(e))
                return {"query": "", "docs": []}

        retriever_chain = RunnableSequence([
            {
                "llm_output": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["llm_output"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the web search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                    }

                reranked_docs = await self.rerank_docs(retriever_result["query"], retriever_result["docs"])
                context = await self.process_docs(reranked_docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"],
                    "context": context
                })

                return {
                    "type": "response",
                    "data": response.content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in reranked_docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while processing the web search results"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are also an expert at summarizing web pages or documents and searching for content in them.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from a search engine and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_web_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a web search query and generate streaming responses.

    Args:
        query: The web search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = WebSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result

<environment_details>
# VSCode Visible Files
backend/agents/academic_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me read the TypeScript version of the web search agent to compare them.
</thinking>

<read_file>
<path>src/agents/webSearchAgent.ts</path>
</read_file>

---

**User:**

[read_file for 'src/agents/webSearchAgent.ts'] Result:
import { BaseMessage } from '@langchain/core/messages';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import { searchSearxng } from '../lib/searxng';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import computeSimilarity from '../utils/computeSimilarity';
import logger from '../utils/logger';
import LineListOutputParser from '../lib/outputParsers/listLineOutputParser';
import { getDocumentsFromLinks } from '../lib/linkDocument';
import LineOutputParser from '../lib/outputParsers/lineOutputParser';
import { IterableReadableStream } from '@langchain/core/utils/stream';
import { ChatOpenAI } from '@langchain/openai';

const basicSearchRetrieverPrompt = `
You are an AI question rephraser. You will be given a conversation and a follow-up question,  you will have to rephrase the follow up question so it is a standalone question and can be used by another LLM to search the web for information to answer it.
If it is a smple writing task or a greeting (unless the greeting contains a question after it) like Hi, Hello, How are you, etc. than a question then you need to return \`not_needed\` as the response (This is because the LLM won't need to search the web for finding information on this topic).
If the user asks some question from some URL or wants you to summarize a PDF or a webpage (via URL) you need to return the links inside the \`links\` XML block and the question inside the \`question\` XML block. If the user wants to you to summarize the webpage or the PDF you need to return \`summarize\` inside the \`question\` XML block in place of a question and the link to summarize in the \`links\` XML block.
You must always return the rephrased question inside the \`question\` XML block, if there are no links in the follow-up question then don't insert a \`links\` XML block in your response.

There are several examples attached for your reference inside the below \`examples\` XML block

<examples>
1. Follow up question: What is the capital of France
Rephrased question:\`
<question>
Capital of france
</question>
\`

2. Hi, how are you?
Rephrased question\`
<question>
not_needed
</question>
\`

3. Follow up question: What is Docker?
Rephrased question: \`
<question>
What is Docker
</question>
\`

4. Follow up question: Can you tell me what is X from https://example.com
Rephrased question: \`
<question>
Can you tell me what is X?
</question>

<links>
https://example.com
</links>
\`

5. Follow up question: Summarize the content from https://example.com
Rephrased question: \`
<question>
summarize
</question>

<links>
https://example.com
</links>
\`
</examples>

Anything below is the part of the actual conversation and you need to use conversation and the follow-up question to rephrase the follow-up question as a standalone question based on the guidelines shared above.

<conversation>
{chat_history}
</conversation>

Follow up question: {query}
Rephrased question:
`;

const basicWebSearchResponsePrompt = `
    You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are also an expert at summarizing web pages or documents and searching for content in them.

    Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
    You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
    You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
    If the query contains some links and the user asks to answer from those links you will be provided the entire content of the page inside the \`context\` XML block. You can then use this content to answer the user's query.
    If the user asks to summarize content from some links, you will be provided the entire content of the page inside the \`context\` XML block. You can then use this content to summarize the text. The content provided inside the \`context\` block will be already summarized by another model so you just need to use that content to answer the user's query.
    Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
    You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
    Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
    However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

    Anything inside the following \`context\` HTML block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to
    talk about the context in your response.

    <context>
    {context}
    </context>

    If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'. You do not need to do this for summarization tasks.
    Anything between the \`context\` is retrieved from a search engine and is not a part of the conversation with the user. Today's date is ${new Date().toISOString()}
`;

const strParser = new StringOutputParser();

const handleStream = async (
  stream: IterableReadableStream<StreamEvent>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalSourceRetriever'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'sources', data: event.data.output }),
      );
    }
    if (
      event.event === 'on_chain_stream' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

type BasicChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const createBasicWebSearchRetrieverChain = (llm: BaseChatModel) => {
  (llm as unknown as ChatOpenAI).temperature = 0;

  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      const linksOutputParser = new LineListOutputParser({
        key: 'links',
      });

      const questionOutputParser = new LineOutputParser({
        key: 'question',
      });

      const links = await linksOutputParser.parse(input);
      let question = await questionOutputParser.parse(input);

      if (question === 'not_needed') {
        return { query: '', docs: [] };
      }

      if (links.length > 0) {
        if (question.length === 0) {
          question = 'summarize';
        }

        let docs = [];

        const linkDocs = await getDocumentsFromLinks({ links });

        const docGroups: Document[] = [];

        linkDocs.map((doc) => {
          const URLDocExists = docGroups.find(
            (d) =>
              d.metadata.url === doc.metadata.url && d.metadata.totalDocs < 10,
          );

          if (!URLDocExists) {
            docGroups.push({
              ...doc,
              metadata: {
                ...doc.metadata,
                totalDocs: 1,
              },
            });
          }

          const docIndex = docGroups.findIndex(
            (d) =>
              d.metadata.url === doc.metadata.url && d.metadata.totalDocs < 10,
          );

          if (docIndex !== -1) {
            docGroups[docIndex].pageContent =
              docGroups[docIndex].pageContent + `\n\n` + doc.pageContent;
            docGroups[docIndex].metadata.totalDocs += 1;
          }
        });

        await Promise.all(
          docGroups.map(async (doc) => {
            const res = await llm.invoke(`
            You are a web search summarizer, tasked with summarizing a piece of text retrieved from a web search. Your job is to summarize the 
            text into a detailed, 2-4 paragraph explanation that captures the main ideas and provides a comprehensive answer to the query.
            If the query is \"summarize\", you should provide a detailed summary of the text. If the query is a specific question, you should answer it in the summary.
            
            - **Journalistic tone**: The summary should sound professional and journalistic, not too casual or vague.
            - **Thorough and detailed**: Ensure that every key point from the text is captured and that the summary directly answers the query.
            - **Not too lengthy, but detailed**: The summary should be informative but not excessively long. Focus on providing detailed information in a concise format.

            The text will be shared inside the \`text\` XML tag, and the query inside the \`query\` XML tag.

            <example>
            <text>
            Docker is a set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers. 
            It was first released in 2013 and is developed by Docker, Inc. Docker is designed to make it easier to create, deploy, and run applications 
            by using containers.
            </text>

            <query>
            What is Docker and how does it work?
            </query>

            Response:
            Docker is a revolutionary platform-as-a-service product developed by Docker, Inc., that uses container technology to make application 
            deployment more efficient. It allows developers to package their software with all necessary dependencies, making it easier to run in 
            any environment. Released in 2013, Docker has transformed the way applications are built, deployed, and managed.
            </example>

            Everything below is the actual data you will be working with. Good luck!

            <query>
            ${question}
            </query>

            <text>
            ${doc.pageContent}
            </text>

            Make sure to answer the query in the summary.
          `);

            const document = new Document({
              pageContent: res.content as string,
              metadata: {
                title: doc.metadata.title,
                url: doc.metadata.url,
              },
            });

            docs.push(document);
          }),
        );

        return { query: question, docs: docs };
      } else {
        const res = await searchSearxng(question, {
          language: 'en',
        });

        const documents = res.results.map(
          (result) =>
            new Document({
              pageContent: result.content,
              metadata: {
                title: result.title,
                url: result.url,
                ...(result.img_src && { img_src: result.img_src }),
              },
            }),
        );

        return { query: question, docs: documents };
      }
    }),
  ]);
};

const createBasicWebSearchAnsweringChain = (
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const basicWebSearchRetrieverChain = createBasicWebSearchRetrieverChain(llm);

  const processDocs = async (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
  };

  const rerankDocs = async ({
    query,
    docs,
  }: {
    query: string;
    docs: Document[];
  }) => {
    if (docs.length === 0) {
      return docs;
    }

    if (query.toLocaleLowerCase() === 'summarize') {
      return docs;
    }

    const docsWithContent = docs.filter(
      (doc) => doc.pageContent && doc.pageContent.length > 0,
    );

    if (optimizationMode === 'speed') {
      return docsWithContent.slice(0, 15);
    } else if (optimizationMode === 'balanced') {
      const [docEmbeddings, queryEmbedding] = await Promise.all([
        embeddings.embedDocuments(
          docsWithContent.map((doc) => doc.pageContent),
        ),
        embeddings.embedQuery(query),
      ]);

      const similarity = docEmbeddings.map((docEmbedding, i) => {
        const sim = computeSimilarity(queryEmbedding, docEmbedding);

        return {
          index: i,
          similarity: sim,
        };
      });

      const sortedDocs = similarity
        .filter((sim) => sim.similarity > 0.3)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 15)
        .map((sim) => docsWithContent[sim.index]);

      return sortedDocs;
    }
  };

  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: BasicChainInput) => input.query,
      chat_history: (input: BasicChainInput) => input.chat_history,
      context: RunnableSequence.from([
        (input) => ({
          query: input.query,
          chat_history: formatChatHistoryAsString(input.chat_history),
        }),
        basicWebSearchRetrieverChain
          .pipe(rerankDocs)
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicWebSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicWebSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = new eventEmitter();

  try {
    const basicWebSearchAnsweringChain = createBasicWebSearchAnsweringChain(
      llm,
      embeddings,
      optimizationMode,
    );

    const stream = basicWebSearchAnsweringChain.streamEvents(
      {
        chat_history: history,
        query: query,
      },
      {
        version: 'v1',
      },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
      'error',
      JSON.stringify({ data: 'An error has occurred please try again later' }),
    );
    logger.error(`Error in websearch: ${err}`);
  }

  return emitter;
};

const handleWebSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = basicWebSearch(
    message,
    history,
    llm,
    embeddings,
    optimizationMode,
  );
  return emitter;
};

export default handleWebSearch;

<environment_details>
# VSCode Visible Files
backend/agents/academic_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
After comparing the web search agents, I notice several key differences that need to be synchronized:

1. The TypeScript version has more detailed examples in the retriever prompt
2. The TypeScript version has additional instructions in the response prompt about handling links and summarization
3. The TypeScript version has a more sophisticated reranking process with a similarity threshold
4. The TypeScript version explicitly sets temperature to 0 for the retriever

Let me update the Python web search agent to match these improvements.
</thinking>

<write_to_file>
<path>backend/agents/web_search_agent.py</path>
<content>"""
Web Search Agent Module

This module provides functionality for performing web searches, rephrasing queries,
and generating responses based on search results or webpage content. It includes
specialized handling for webpage summarization and link-based content retrieval.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from lib.searxng import search_searxng
from lib.link_document import get_documents_from_links
from utils.logger import logger
from lib.output_parsers.line_output_parser import LineOutputParser
from lib.output_parsers.list_line_output_parser import ListLineOutputParser
from agents.abstract_agent import AbstractAgent

BASIC_SEARCH_RETRIEVER_PROMPT = """
You are an AI question rephraser. You will be given a conversation and a follow-up question,  you will have to rephrase the follow up question so it is a standalone question and can be used by another LLM to search the web for information to answer it.
If it is a smple writing task or a greeting (unless the greeting contains a question after it) like Hi, Hello, How are you, etc. than a question then you need to return `not_needed` as the response (This is because the LLM won't need to search the web for finding information on this topic).
If the user asks some question from some URL or wants you to summarize a PDF or a webpage (via URL) you need to return the links inside the `links` XML block and the question inside the `question` XML block. If the user wants to you to summarize the webpage or the PDF you need to return `summarize` inside the `question` XML block in place of a question and the link to summarize in the `links` XML block.
You must always return the rephrased question inside the `question` XML block, if there are no links in the follow-up question then don't insert a `links` XML block in your response.

There are several examples attached for your reference inside the below `examples` XML block

<examples>
1. Follow up question: What is the capital of France
Rephrased question:`
<question>
Capital of france
</question>
`

2. Hi, how are you?
Rephrased question`
<question>
not_needed
</question>
`

3. Follow up question: What is Docker?
Rephrased question: `
<question>
What is Docker
</question>
`

4. Follow up question: Can you tell me what is X from https://example.com
Rephrased question: `
<question>
Can you tell me what is X?
</question>

<links>
https://example.com
</links>
`

5. Follow up question: Summarize the content from https://example.com
Rephrased question: `
<question>
summarize
</question>

<links>
https://example.com
</links>
`
</examples>

Anything below is the part of the actual conversation and you need to use conversation and the follow-up question to rephrase the follow-up question as a standalone question based on the guidelines shared above.

<conversation>
{chat_history}
</conversation>

Follow up question: {query}
Rephrased question:
"""

class WebSearchAgent(AbstractAgent):
    """Agent for performing web searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the web search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_SEARCH_RETRIEVER_PROMPT)

        # Set temperature to 0 for more precise rephrasing
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = 0

        chain = prompt | self.llm

        async def process_output(llm_output: Any) -> Dict[str, Any]:
            """Process the LLM output and perform the web search."""
            try:
                output_text = llm_output.content if hasattr(llm_output, 'content') else str(llm_output)
                links_parser = ListLineOutputParser(key="links")
                question_parser = LineOutputParser(key="question")

                links = await links_parser.parse(output_text)
                question = await question_parser.parse(output_text)

                if question == "not_needed":
                    return {"query": "", "docs": []}

                if links:
                    if not question:
                        question = "summarize"

                    docs = []
                    link_docs = await get_documents_from_links({"links": links})
                    doc_groups = []

                    for doc in link_docs:
                        url_doc = next(
                            (d for d in doc_groups if d.metadata["url"] == doc.metadata["url"] and d.metadata.get("totalDocs", 0) < 10),
                            None
                        )

                        if not url_doc:
                            doc.metadata["totalDocs"] = 1
                            doc_groups.append(doc)
                        else:
                            url_doc.page_content += f"\n\n{doc.page_content}"
                            url_doc.metadata["totalDocs"] = url_doc.metadata.get("totalDocs", 0) + 1

                    for doc in doc_groups:
                        try:
                            summary_prompt = ChatPromptTemplate.from_messages([
                                ("system", """
                                You are a web search summarizer, tasked with summarizing a piece of text retrieved from a web search. Your job is to summarize the 
                                text into a detailed, 2-4 paragraph explanation that captures the main ideas and provides a comprehensive answer to the query.
                                If the query is "summarize", you should provide a detailed summary of the text. If the query is a specific question, you should answer it in the summary.

                                - **Journalistic tone**: The summary should sound professional and journalistic, not too casual or vague.
                                - **Thorough and detailed**: Ensure that every key point from the text is captured and that the summary directly answers the query.
                                - **Not too lengthy, but detailed**: The summary should be informative but not excessively long. Focus on providing detailed information in a concise format.

                                The text will be shared inside the `text` XML tag, and the query inside the `query` XML tag.

                                <example>
                                <text>
                                Docker is a set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers. 
                                It was first released in 2013 and is developed by Docker, Inc. Docker is designed to make it easier to create, deploy, and run applications 
                                by using containers.
                                </text>

                                <query>
                                What is Docker and how does it work?
                                </query>

                                Response:
                                Docker is a revolutionary platform-as-a-service product developed by Docker, Inc., that uses container technology to make application 
                                deployment more efficient. It allows developers to package their software with all necessary dependencies, making it easier to run in 
                                any environment. Released in 2013, Docker has transformed the way applications are built, deployed, and managed.
                                </example>

                                Everything below is the actual data you will be working with. Good luck!

                                <query>
                                {query}
                                </query>

                                <text>
                                {text}
                                </text>

                                Make sure to answer the query in the summary.
                                """),
                                ("human", "{query}")
                            ])

                            summary_chain = summary_prompt | self.llm

                            summary = await summary_chain.invoke({
                                "query": question,
                                "text": doc.page_content
                            })

                            docs.append(Document(
                                page_content=summary.content,
                                metadata={
                                    "title": doc.metadata["title"],
                                    "url": doc.metadata["url"]
                                }
                            ))
                        except Exception as e:
                            logger.error("Error summarizing document: %s", str(e))
                            docs.append(doc)

                    return {"query": question, "docs": docs}
                else:
                    res = await search_searxng(question, {
                        "language": "en"
                    })

                    documents = [
                        Document(
                            page_content=result["content"],
                            metadata={
                                "title": result["title"],
                                "url": result["url"],
                                **({"img_src": result["img_src"]} if "img_src" in result else {})
                            }
                        )
                        for result in res["results"]
                    ]

                    return {"query": question, "docs": documents}
            except Exception as e:
                logger.error("Error in web search/processing: %s", str(e))
                return {"query": "", "docs": []}

        retriever_chain = RunnableSequence([
            {
                "llm_output": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["llm_output"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the web search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                    }

                reranked_docs = await self.rerank_docs(retriever_result["query"], retriever_result["docs"])
                context = await self.process_docs(reranked_docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"],
                    "context": context
                })

                return {
                    "type": "response",
                    "data": response.content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in reranked_docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while processing the web search results"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are also an expert at summarizing web pages or documents and searching for content in them.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        If the query contains some links and the user asks to answer from those links you will be provided the entire content of the page inside the `context` XML block. You can then use this content to answer the user's query.
        If the user asks to summarize content from some links, you will be provided the entire content of the page inside the `context` XML block. You can then use this content to summarize the text. The content provided inside the `context` block will be already summarized by another model so you just need to use that content to answer the user's query.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
        However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

        Anything inside the following `context` HTML block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to
        talk about the context in your response.

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'. You do not need to do this for summarization tasks.
        Anything between the `context` is retrieved from a search engine and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query."""
        if not docs:
            return docs

        if query.lower() == "summarize":
            return docs

        docs_with_content = [doc for doc in docs if doc.page_content and len(doc.page_content) > 0]

        if self.optimization_mode == "speed":
            return docs_with_content[:15]
        elif self.optimization_mode == "balanced":
            doc_embeddings = await self.embeddings.embed_documents([doc.page_content for doc in docs_with_content])
            query_embedding = await self.embeddings.embed_query(query)

            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                from utils.compute_similarity import compute_similarity
                sim = compute_similarity(query_embedding, doc_embedding)
                similarities.append({"index": i, "similarity": sim})

            # Filter docs with similarity > 0.3 and take top 15
            sorted_docs = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
            filtered_docs = [docs_with_content[s["index"]] for s in sorted_docs if s["similarity"] > 0.3][:15]
            return filtered_docs

        return docs

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_web_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a web search query and generate streaming responses.

    Args:
        query: The web search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = WebSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
</content>
</write_to_file>

---

**User:**

[write_to_file for 'backend/agents/web_search_agent.py'] Result:
The content was successfully saved to backend/agents/web_search_agent.py.
<environment_details>
# VSCode Visible Files
backend/agents/web_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me check another important agent pair. I'll look at the writing assistant agent next since it's a core functionality.
</thinking>

<read_file>
<path>backend/agents/writing_assistant_agent.py</path>
</read_file>

---

**User:**

[read_file for 'backend/agents/writing_assistant_agent.py'] Result:
"""
Writing Assistant Agent Module

This module provides functionality for assisting users with their writing tasks.
It focuses on providing direct writing assistance without performing web searches.
"""

from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

WRITING_ASSISTANT_PROMPT = """
You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are currently set on focus mode 'Writing Assistant', this means you will be helping the user write a response to a given query.
Since you are a writing assistant, you would not perform web searches. If you think you lack information to answer the query, you can ask the user for more information or suggest them to switch to a different focus mode.
"""

class WritingAssistantAgent(AbstractAgent):
    """Agent for providing writing assistance without web searches."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the writing assistant retriever chain."""
        async def process_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the input and return empty docs since we don't need retrieval."""
            return {"query": input_data["query"], "docs": []}

        return RunnableSequence([process_input])

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the writing assistant answering chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", WRITING_ASSISTANT_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])

        chain = prompt | self.llm
        str_parser = StrOutputParser()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate a response to the user's writing-related query."""
            try:
                response = await chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                return {
                    "type": "response",
                    "data": str_parser.parse(response.content)
                }
            except Exception as e:
                logger.error("Error in writing assistant chain: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while processing your writing request"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        # Not used as we use a fixed prompt template in create_answering_chain
        return WRITING_ASSISTANT_PROMPT

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_writing_assistant(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a writing assistance request and generate streaming responses.

    Args:
        query: The user's writing-related query or request
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: Not used for writing assistance but required by base class
        optimization_mode: Not used for writing assistance but required by base class

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = WritingAssistantAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result

<environment_details>
# VSCode Visible Files
backend/agents/web_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me read the TypeScript version of the writing assistant agent to compare them.
</thinking>

<read_file>
<path>src/agents/writingAssistant.ts</path>
</read_file>

---

**User:**

[read_file for 'src/agents/writingAssistant.ts'] Result:
import { BaseMessage } from '@langchain/core/messages';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import eventEmitter from 'events';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import logger from '../utils/logger';
import { IterableReadableStream } from '@langchain/core/utils/stream';

const writingAssistantPrompt = `
You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are currently set on focus mode 'Writing Assistant', this means you will be helping the user write a response to a given query. 
Since you are a writing assistant, you would not perform web searches. If you think you lack information to answer the query, you can ask the user for more information or suggest them to switch to a different focus mode.
`;

const strParser = new StringOutputParser();

const handleStream = async (
  stream: IterableReadableStream<StreamEvent>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
      event.event === 'on_chain_stream' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

const createWritingAssistantChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    ChatPromptTemplate.fromMessages([
      ['system', writingAssistantPrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const handleWritingAssistant = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = new eventEmitter();

  try {
    const writingAssistantChain = createWritingAssistantChain(llm);
    const stream = writingAssistantChain.streamEvents(
      {
        chat_history: history,
        query: query,
      },
      {
        version: 'v1',
      },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
      'error',
      JSON.stringify({ data: 'An error has occurred please try again later' }),
    );
    logger.error(`Error in writing assistant: ${err}`);
  }

  return emitter;
};

export default handleWritingAssistant;

<environment_details>
# VSCode Visible Files
backend/agents/web_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
After comparing the writing assistant implementations, I notice they are mostly in sync. The main differences are:

1. The Python version has a more complex class-based structure inherited from AbstractAgent
2. The TypeScript version is more streamlined, focusing only on the core writing assistance functionality
3. Both use the same prompt and basic approach

Since the Python version needs to follow the AbstractAgent interface but maintains the same core functionality, I don't see any necessary changes to sync between them. The additional structure in Python is required by its architecture but doesn't affect the actual functionality.

Let me check one more agent pair to be thorough. I'll look at the suggestion generator agent since it's another important one.
</thinking>

<read_file>
<path>backend/agents/suggestion_generator_agent.py</path>
</read_file>

---

**User:**

[read_file for 'backend/agents/suggestion_generator_agent.py'] Result:
"""
Suggestion Generator Agent Module

This module provides functionality for generating relevant suggestions based on chat history.
It analyzes the conversation context to provide meaningful follow-up questions or topics.
"""

from typing import List, Dict, Any, Literal, AsyncGenerator, Optional
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from utils.format_history import format_chat_history_as_string
from lib.output_parsers.list_line_output_parser import ListLineOutputParser
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

SUGGESTION_GENERATOR_PROMPT = """
You are an AI suggestion generator for an AI powered search engine. You will be given a conversation below. You need to generate 4-5 suggestions based on the conversation. The suggestion should be relevant to the conversation that can be used by the user to ask the chat model for more information.
You need to make sure the suggestions are relevant to the conversation and are helpful to the user. Keep a note that the user might use these suggestions to ask a chat model for more information.
Make sure the suggestions are medium in length and are informative and relevant to the conversation.

Provide these suggestions separated by newlines between the XML tags <suggestions> and </suggestions>. For example:

<suggestions>
Tell me more about SpaceX and their recent projects
What is the latest news on SpaceX?
Who is the CEO of SpaceX?
</suggestions>

Conversation:
{chat_history}
"""

class SuggestionGeneratorAgent(AbstractAgent):
    """Agent for generating suggestions based on chat history."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the suggestion generator retriever chain."""
        async def process_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the input and return empty docs since we don't need retrieval."""
            return {"query": "", "docs": []}

        return RunnableSequence([process_input])

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the suggestion generator answering chain."""
        # Set temperature to 0 for more consistent suggestions
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = 0

        prompt = PromptTemplate.from_template(SUGGESTION_GENERATOR_PROMPT)
        output_parser = ListLineOutputParser(key="suggestions")

        async def map_input(input_data: Dict[str, Any]) -> Dict[str, str]:
            """Map the input data to the format expected by the prompt."""
            return {
                "chat_history": format_chat_history_as_string(input_data["chat_history"])
            }

        async def process_output(output: str) -> Dict[str, Any]:
            """Process the output into the expected format."""
            try:
                suggestions = await output_parser.parse(output)
                return {
                    "type": "response",
                    "data": suggestions
                }
            except Exception as e:
                logger.error("Error parsing suggestions: %s", str(e))
                return {
                    "type": "response",
                    "data": ["Unable to generate suggestions at this time"]
                }

        chain = RunnableSequence([
            map_input,
            prompt | self.llm,
            lambda x: x.content if hasattr(x, 'content') else x,
            process_output
        ])

        return chain

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        # Not used for suggestion generation as we use a fixed prompt
        return SUGGESTION_GENERATOR_PROMPT

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response.split("\n") if response else []
        }

async def generate_suggestions(
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Optional[Embeddings] = None,  # Not used but kept for interface compatibility
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"  # Not used but kept for interface compatibility
) -> List[str]:
    """
    Generate suggestions based on chat history.

    Args:
        history: The chat history to generate suggestions from
        llm: The language model to use for generating suggestions
        embeddings: Not used for suggestion generation
        optimization_mode: Not used for suggestion generation

    Returns:
        List[str]: A list of generated suggestions. Returns a single error message
                  if suggestion generation fails.
    """
    try:
        # Create a dummy embeddings object if none provided since AbstractAgent requires it
        if embeddings is None:
            class DummyEmbeddings(Embeddings):
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.0] for _ in texts]
                def embed_query(self, text: str) -> List[float]:
                    return [0.0]
            embeddings = DummyEmbeddings()

        agent = SuggestionGeneratorAgent(llm, embeddings, optimization_mode)
        async for result in agent.handle_search("", history):  # Empty query since we only need history
            if result["type"] == "response":
                return result["data"]
        return ["Unable to generate suggestions at this time"]
    except Exception as e:
        logger.error("Error in generate_suggestions: %s", str(e))
        return ["Unable to generate suggestions at this time"]

<environment_details>
# VSCode Visible Files
backend/agents/web_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me read the TypeScript version of the suggestion generator agent to compare them.
</thinking>

<read_file>
<path>src/agents/suggestionGeneratorAgent.ts</path>
</read_file>

---

**User:**

[read_file for 'src/agents/suggestionGeneratorAgent.ts'] Result:
import { RunnableSequence, RunnableMap } from '@langchain/core/runnables';
import ListLineOutputParser from '../lib/outputParsers/listLineOutputParser';
import { PromptTemplate } from '@langchain/core/prompts';
import formatChatHistoryAsString from '../utils/formatHistory';
import { BaseMessage } from '@langchain/core/messages';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { ChatOpenAI } from '@langchain/openai';

const suggestionGeneratorPrompt = `
You are an AI suggestion generator for an AI powered search engine. You will be given a conversation below. You need to generate 4-5 suggestions based on the conversation. The suggestion should be relevant to the conversation that can be used by the user to ask the chat model for more information.
You need to make sure the suggestions are relevant to the conversation and are helpful to the user. Keep a note that the user might use these suggestions to ask a chat model for more information. 
Make sure the suggestions are medium in length and are informative and relevant to the conversation.

Provide these suggestions separated by newlines between the XML tags <suggestions> and </suggestions>. For example:

<suggestions>
Tell me more about SpaceX and their recent projects
What is the latest news on SpaceX?
Who is the CEO of SpaceX?
</suggestions>

Conversation:
{chat_history}
`;

type SuggestionGeneratorInput = {
  chat_history: BaseMessage[];
};

const outputParser = new ListLineOutputParser({
  key: 'suggestions',
});

const createSuggestionGeneratorChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    RunnableMap.from({
      chat_history: (input: SuggestionGeneratorInput) =>
        formatChatHistoryAsString(input.chat_history),
    }),
    PromptTemplate.fromTemplate(suggestionGeneratorPrompt),
    llm,
    outputParser,
  ]);
};

const generateSuggestions = (
  input: SuggestionGeneratorInput,
  llm: BaseChatModel,
) => {
  (llm as unknown as ChatOpenAI).temperature = 0;
  const suggestionGeneratorChain = createSuggestionGeneratorChain(llm);
  return suggestionGeneratorChain.invoke(input);
};

export default generateSuggestions;

<environment_details>
# VSCode Visible Files
backend/agents/web_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
After comparing all the agent pairs, here's a summary of my findings:

1. Academic Search Agent:
   - Updated Python version with more detailed citation instructions and context handling from TypeScript version

2. Web Search Agent:
   - Updated Python version with more detailed examples, similarity threshold for reranking, and explicit temperature setting

3. Writing Assistant Agent:
   - Both versions are functionally equivalent
   - Python version has more structure due to AbstractAgent requirements
   - No changes needed

4. Suggestion Generator Agent:
   - Both versions use the same prompt and approach
   - Both set temperature to 0
   - Python version has additional error handling and interface compatibility
   - No changes needed

The main differences have been addressed in the updates I made to the academic and web search agents. The writing and suggestion generator agents are already well-synchronized in terms of functionality, with their structural differences being necessary due to the Python codebase's architecture requirements.

<thinking>
Let me verify if there are any other agents that need comparison.
</thinking>

<list_files>
<path>src/agents</path>
</list_files>

---

**User:**

[list_files for 'src/agents'] Result:
academicSearchAgent.ts
imageSearchAgent.ts
redditSearchAgent.ts
suggestionGeneratorAgent.ts
videoSearchAgent.ts
webSearchAgent.ts
wolframAlphaSearchAgent.ts
writingAssistant.ts
youtubeSearchAgent.ts
<environment_details>
# VSCode Visible Files
backend/agents/web_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
I should check the remaining agents to ensure complete synchronization. Let me start with the image search agent.
</thinking>

<read_file>
<path>backend/agents/image_search_agent.py</path>
</read_file>

---

**User:**

[read_file for 'backend/agents/image_search_agent.py'] Result:
from typing import AsyncGenerator, Dict, Any
from langchain.schema import HumanMessage, AIMessage

from lib.searxng import search_searxng

async def handle_image_search(
    query: str,
    history: list[HumanMessage | AIMessage],
    chat_model: Any,
    embeddings_model: Any,
    mode: str = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle image search requests.
    
    Args:
        query: The search query
        history: Chat history
        chat_model: The chat model to use
        embeddings_model: The embeddings model to use
        mode: Search mode (speed/balanced/quality)
        
    Yields:
        Dict containing response type and data
    """
    try:
        # Perform image search
        results = await search_searxng(query, "images")

        if not results or not results.get("results"):
            yield {
                "type": "error",
                "data": "No images found. Please try a different search query."
            }
            return

        # Format image results
        image_results = []
        for result in results["results"]:
            if "img_src" in result:
                image_results.append({
                    "url": result["img_src"],
                    "title": result.get("title", ""),
                    "source": result.get("source", "")
                })

        if not image_results:
            yield {
                "type": "error",
                "data": "No valid images found. Please try a different search query."
            }
            return

        yield {
            "type": "images",
            "data": image_results
        }

    except Exception as e:
        yield {
            "type": "error",
            "data": f"An error occurred during image search: {str(e)}"
        }

<environment_details>
# VSCode Visible Files
backend/agents/web_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me read the TypeScript version of the image search agent to compare them.
</thinking>

<read_file>
<path>src/agents/imageSearchAgent.ts</path>
</read_file>

---

**User:**

[read_file for 'src/agents/imageSearchAgent.ts'] Result:
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import formatChatHistoryAsString from '../utils/formatHistory';
import { BaseMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { searchSearxng } from '../lib/searxng';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';

const imageSearchChainPrompt = `
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question so it is a standalone question that can be used by the LLM to search the web for images.
You need to make sure the rephrased question agrees with the conversation and is relevant to the conversation.

Example:
1. Follow up question: What is a cat?
Rephrased: A cat

2. Follow up question: What is a car? How does it works?
Rephrased: Car working

3. Follow up question: How does an AC work?
Rephrased: AC working

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
`;

type ImageSearchChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const strParser = new StringOutputParser();

const createImageSearchChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    RunnableMap.from({
      chat_history: (input: ImageSearchChainInput) => {
        return formatChatHistoryAsString(input.chat_history);
      },
      query: (input: ImageSearchChainInput) => {
        return input.query;
      },
    }),
    PromptTemplate.fromTemplate(imageSearchChainPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      const res = await searchSearxng(input, {
        engines: ['bing images', 'google images'],
      });

      const images = [];

      res.results.forEach((result) => {
        if (result.img_src && result.url && result.title) {
          images.push({
            img_src: result.img_src,
            url: result.url,
            title: result.title,
          });
        }
      });

      return images.slice(0, 10);
    }),
  ]);
};

const handleImageSearch = (
  input: ImageSearchChainInput,
  llm: BaseChatModel,
) => {
  const imageSearchChain = createImageSearchChain(llm);
  return imageSearchChain.invoke(input);
};

export default handleImageSearch;

<environment_details>
# VSCode Visible Files
backend/agents/web_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
I notice several differences between the image search agents:

1. The TypeScript version has a query rephrasing step using a prompt, while the Python version searches directly
2. The TypeScript version specifies specific image search engines (bing images, google images)
3. The TypeScript version limits results to 10 images
4. The TypeScript version has a more structured chain-based approach

Let me update the Python version to match these improvements.
</thinking>

<write_to_file>
<path>backend/agents/image_search_agent.py</path>
<content>"""
Image Search Agent Module

This module provides functionality for performing image searches, with query rephrasing
and result processing to ensure relevant and high-quality image results.
"""

from typing import AsyncGenerator, Dict, Any, List
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.format_history import format_chat_history_as_string
from utils.logger import logger

IMAGE_SEARCH_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question so it is a standalone question that can be used by the LLM to search the web for images.
You need to make sure the rephrased question agrees with the conversation and is relevant to the conversation.

Example:
1. Follow up question: What is a cat?
Rephrased: A cat

2. Follow up question: What is a car? How does it works?
Rephrased: Car working

3. Follow up question: How does an AC work?
Rephrased: AC working

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

async def handle_image_search(
    query: str,
    history: List[HumanMessage | AIMessage],
    chat_model: Any,
    embeddings_model: Any,
    mode: str = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle image search requests.
    
    Args:
        query: The search query
        history: Chat history
        chat_model: The chat model to use
        embeddings_model: The embeddings model to use
        mode: Search mode (speed/balanced/quality)
        
    Yields:
        Dict containing response type and data
    """
    try:
        # Create and run the query rephrasing chain
        prompt = PromptTemplate.from_template(IMAGE_SEARCH_PROMPT)
        str_parser = StrOutputParser()
        
        # Format chat history
        formatted_history = format_chat_history_as_string(history)
        
        # Get rephrased query
        chain_response = await chat_model.invoke(
            prompt.format(chat_history=formatted_history, query=query)
        )
        rephrased_query = str_parser.parse(chain_response.content)

        # Perform image search with specific engines
        results = await search_searxng(rephrased_query, {
            "engines": ["bing images", "google images"]
        })

        if not results or not results.get("results"):
            yield {
                "type": "error",
                "data": "No images found. Please try a different search query."
            }
            return

        # Format image results
        image_results = []
        for result in results["results"]:
            if all(key in result for key in ["img_src", "url", "title"]):
                image_results.append({
                    "img_src": result["img_src"],
                    "url": result["url"],
                    "title": result["title"]
                })

        # Limit to 10 results
        image_results = image_results[:10]

        if not image_results:
            yield {
                "type": "error",
                "data": "No valid images found. Please try a different search query."
            }
            return

        yield {
            "type": "images",
            "data": image_results
        }

    except Exception as e:
        logger.error("Error in image search: %s", str(e))
        yield {
            "type": "error",
            "data": f"An error occurred during image search: {str(e)}"
        }
</content>
</write_to_file>

---

**User:**

[write_to_file for 'backend/agents/image_search_agent.py'] Result:
The content was successfully saved to backend/agents/image_search_agent.py.
<environment_details>
# VSCode Visible Files
backend/agents/image_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me check the video search agent next to ensure it's properly synchronized.
</thinking>

<read_file>
<path>backend/agents/video_search_agent.py</path>
</read_file>

---

**User:**

[read_file for 'backend/agents/video_search_agent.py'] Result:
"""
Video Search Agent Module

This module provides functionality for performing video searches using the SearxNG search engine
with YouTube as the primary source. It specifically handles video content with support for
thumbnails and embedded iframes.
"""

from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

VIDEO_SEARCH_CHAIN_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question so it is a standalone question that can be used by the LLM to search Youtube for videos.
You need to make sure the rephrased question agrees with the conversation and is relevant to the conversation.

Example:
1. Follow up question: How does a car work?
Rephrased: How does a car work?

2. Follow up question: What is the theory of relativity?
Rephrased: What is theory of relativity

3. Follow up question: How does an AC work?
Rephrased: How does an AC work

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class VideoSearchAgent(AbstractAgent):
    """Agent for performing video searches and retrieving results."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the video search retriever chain."""
        prompt = PromptTemplate.from_template(VIDEO_SEARCH_CHAIN_PROMPT)
        str_parser = StrOutputParser()

        chain = prompt | self.llm | str_parser

        async def perform_search(query: str) -> Dict[str, Any]:
            """Perform the video search using YouTube through SearxNG."""
            try:
                res = await search_searxng(query, {
                    "engines": ["youtube"]
                })

                documents = []
                for result in res["results"]:
                    if (result.get("thumbnail") and
                        result.get("url") and
                        result.get("title") and
                        result.get("iframe_src")):
                        documents.append(Document(
                            page_content="",  # Video searches don't need text content
                            metadata={
                                "img_src": result["thumbnail"],
                                "url": result["url"],
                                "title": result["title"],
                                "iframe_src": result["iframe_src"]
                            }
                        ))

                return {"query": query, "docs": documents[:10]}  # Return top 10 videos

            except Exception as e:
                logger.error("Error in video search: %s", str(e))
                return {"query": query, "docs": []}

        retriever_chain = RunnableSequence([
            {
                "rephrased_query": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: perform_search(x["rephrased_query"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the video search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def process_results(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the retriever results into the final format."""
            try:
                result = await retriever_chain.invoke(input_data)

                if not result["docs"]:
                    return {
                        "type": "response",
                        "data": []
                    }

                # Extract video data from documents
                videos = [
                    {
                        "img_src": doc.metadata["img_src"],
                        "url": doc.metadata["url"],
                        "title": doc.metadata["title"],
                        "iframe_src": doc.metadata["iframe_src"]
                    }
                    for doc in result["docs"]
                ]

                return {
                    "type": "response",
                    "data": videos
                }

            except Exception as e:
                logger.error("Error processing video search results: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while processing the video search results"
                }

        return RunnableSequence([process_results])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        # Not used for video search as it doesn't require text generation
        return ""

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        # Not used for video search as it doesn't require text generation
        return {
            "type": "response",
            "data": response
        }

async def handle_video_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> List[Dict[str, str]]:
    """
    Handle a video search query.

    Args:
        query: The video search query
        history: The chat history
        llm: The language model to use for query processing
        embeddings: The embeddings model (unused for video search)
        optimization_mode: The optimization mode (unused for video search)

    Returns:
        List[Dict[str, str]]: A list of video results, each containing:
            - img_src: URL of the video thumbnail
            - url: URL of the video page
            - title: Title of the video
            - iframe_src: Embedded iframe URL for the video
    """
    agent = VideoSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        if result["type"] == "response":
            return result["data"]
    return []  # Return empty list if no results or error

<environment_details>
# VSCode Visible Files
backend/agents/image_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me read the TypeScript version of the video search agent to compare them.
</thinking>

<read_file>
<path>src/agents/videoSearchAgent.ts</path>
</read_file>

---

**User:**

[read_file for 'src/agents/videoSearchAgent.ts'] Result:
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import formatChatHistoryAsString from '../utils/formatHistory';
import { BaseMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { searchSearxng } from '../lib/searxng';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';

const VideoSearchChainPrompt = `
  You will be given a conversation below and a follow up question. You need to rephrase the follow-up question so it is a standalone question that can be used by the LLM to search Youtube for videos.
  You need to make sure the rephrased question agrees with the conversation and is relevant to the conversation.
  
  Example:
  1. Follow up question: How does a car work?
  Rephrased: How does a car work?
  
  2. Follow up question: What is the theory of relativity?
  Rephrased: What is theory of relativity
  
  3. Follow up question: How does an AC work?
  Rephrased: How does an AC work
  
  Conversation:
  {chat_history}
  
  Follow up question: {query}
  Rephrased question:
  `;

type VideoSearchChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const strParser = new StringOutputParser();

const createVideoSearchChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    RunnableMap.from({
      chat_history: (input: VideoSearchChainInput) => {
        return formatChatHistoryAsString(input.chat_history);
      },
      query: (input: VideoSearchChainInput) => {
        return input.query;
      },
    }),
    PromptTemplate.fromTemplate(VideoSearchChainPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      const res = await searchSearxng(input, {
        engines: ['youtube'],
      });

      const videos = [];

      res.results.forEach((result) => {
        if (
          result.thumbnail &&
          result.url &&
          result.title &&
          result.iframe_src
        ) {
          videos.push({
            img_src: result.thumbnail,
            url: result.url,
            title: result.title,
            iframe_src: result.iframe_src,
          });
        }
      });

      return videos.slice(0, 10);
    }),
  ]);
};

const handleVideoSearch = (
  input: VideoSearchChainInput,
  llm: BaseChatModel,
) => {
  const VideoSearchChain = createVideoSearchChain(llm);
  return VideoSearchChain.invoke(input);
};

export default handleVideoSearch;

<environment_details>
# VSCode Visible Files
backend/agents/image_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
After comparing the video search agents, I notice they are already well-synchronized in terms of core functionality:

1. Both use the same prompt for query rephrasing
2. Both use YouTube as the search engine
3. Both limit results to 10 videos
4. Both check for the same required fields (thumbnail/img_src, url, title, iframe_src)

The main difference is structural:
- Python version uses the AbstractAgent class structure
- TypeScript version uses a more functional approach

Since these differences are architectural requirements of each codebase and don't affect functionality, no changes are needed to synchronize them.

Let me check the reddit search agent next.
</thinking>

<read_file>
<path>backend/agents/reddit_search_agent.py</path>
</read_file>

---

**User:**

[read_file for 'backend/agents/reddit_search_agent.py'] Result:
"""
Reddit Search Agent Module

This module provides functionality for performing Reddit searches using the SearxNG search engine.
It includes document reranking capabilities to prioritize the most relevant Reddit discussions
and comments.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

BASIC_REDDIT_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: Which company is most likely to create an AGI
Rephrased: Which company is most likely to create an AGI

2. Follow up question: Is Earth flat?
Rephrased: Is Earth flat?

3. Follow up question: Is there life on Mars?
Rephrased: Is there life on Mars?

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class RedditSearchAgent(AbstractAgent):
    """Agent for performing Reddit searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the Reddit search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_REDDIT_SEARCH_RETRIEVER_PROMPT)
        str_parser = StrOutputParser()

        chain = prompt | self.llm | str_parser

        async def process_output(query_text: str) -> Dict[str, Any]:
            """Process the LLM output and perform the Reddit search."""
            if query_text.strip() == "not_needed":
                return {"query": "", "docs": []}

            try:
                res = await search_searxng(query_text, {
                    "language": "en",
                    "engines": ["reddit"]
                })

                documents = [
                    Document(
                        page_content=result.get("content", result["title"]),
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            **({"img_src": result["img_src"]} if result.get("img_src") else {})
                        }
                    )
                    for result in res["results"]
                ]

                return {"query": query_text, "docs": documents}
            except Exception as e:
                logger.error("Error in Reddit search: %s", str(e))
                return {"query": query_text, "docs": []}

        retriever_chain = RunnableSequence([
            {
                "rephrased_query": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["rephrased_query"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the Reddit search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                    }

                reranked_docs = await self.rerank_docs(retriever_result["query"], retriever_result["docs"])
                context = await self.process_docs(reranked_docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({"query": input_data["query"]})

                return {
                    "type": "response",
                    "data": response.content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in reranked_docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while processing Reddit search results"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Reddit', this means you will be searching for information, opinions and discussions on the web using Reddit.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from Reddit and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_reddit_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a Reddit search query and generate streaming responses.

    Args:
        query: The Reddit search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = RedditSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result

<environment_details>
# VSCode Visible Files
backend/agents/image_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me read the TypeScript version of the reddit search agent to compare them.
</thinking>

<read_file>
<path>src/agents/redditSearchAgent.ts</path>
</read_file>

---

**User:**

[read_file for 'src/agents/redditSearchAgent.ts'] Result:
import { BaseMessage } from '@langchain/core/messages';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import { searchSearxng } from '../lib/searxng';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import computeSimilarity from '../utils/computeSimilarity';
import logger from '../utils/logger';
import { IterableReadableStream } from '@langchain/core/utils/stream';

const basicRedditSearchRetrieverPrompt = `
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return \`not_needed\` as the response.

Example:
1. Follow up question: Which company is most likely to create an AGI
Rephrased: Which company is most likely to create an AGI

2. Follow up question: Is Earth flat?
Rephrased: Is Earth flat?

3. Follow up question: Is there life on Mars?
Rephrased: Is there life on Mars?

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
`;

const basicRedditSearchResponsePrompt = `
    You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Reddit', this means you will be searching for information, opinions and discussions on the web using Reddit.

    Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
    You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
    You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
    Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
    You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
    Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
    However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

    Anything inside the following \`context\` HTML block provided below is for your knowledge returned by Reddit and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to
    talk about the context in your response.

    <context>
    {context}
    </context>

    If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
    Anything between the \`context\` is retrieved from Reddit and is not a part of the conversation with the user. Today's date is ${new Date().toISOString()}
`;

const strParser = new StringOutputParser();

const handleStream = async (
  stream: IterableReadableStream<StreamEvent>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalSourceRetriever'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'sources', data: event.data.output }),
      );
    }
    if (
      event.event === 'on_chain_stream' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

type BasicChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const createBasicRedditSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicRedditSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'en',
        engines: ['reddit'],
      });

      const documents = res.results.map(
        (result) =>
          new Document({
            pageContent: result.content ? result.content : result.title,
            metadata: {
              title: result.title,
              url: result.url,
              ...(result.img_src && { img_src: result.img_src }),
            },
          }),
      );

      return { query: input, docs: documents };
    }),
  ]);
};

const createBasicRedditSearchAnsweringChain = (
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const basicRedditSearchRetrieverChain =
    createBasicRedditSearchRetrieverChain(llm);

  const processDocs = async (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
  };

  const rerankDocs = async ({
    query,
    docs,
  }: {
    query: string;
    docs: Document[];
  }) => {
    if (docs.length === 0) {
      return docs;
    }

    const docsWithContent = docs.filter(
      (doc) => doc.pageContent && doc.pageContent.length > 0,
    );

    if (optimizationMode === 'speed') {
      return docsWithContent.slice(0, 15);
    } else if (optimizationMode === 'balanced') {
      const [docEmbeddings, queryEmbedding] = await Promise.all([
        embeddings.embedDocuments(
          docsWithContent.map((doc) => doc.pageContent),
        ),
        embeddings.embedQuery(query),
      ]);

      const similarity = docEmbeddings.map((docEmbedding, i) => {
        const sim = computeSimilarity(queryEmbedding, docEmbedding);

        return {
          index: i,
          similarity: sim,
        };
      });

      const sortedDocs = similarity
        .filter((sim) => sim.similarity > 0.3)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 15)
        .map((sim) => docsWithContent[sim.index]);

      return sortedDocs;
    }
  };

  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: BasicChainInput) => input.query,
      chat_history: (input: BasicChainInput) => input.chat_history,
      context: RunnableSequence.from([
        (input) => ({
          query: input.query,
          chat_history: formatChatHistoryAsString(input.chat_history),
        }),
        basicRedditSearchRetrieverChain
          .pipe(rerankDocs)
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicRedditSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicRedditSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = new eventEmitter();

  try {
    const basicRedditSearchAnsweringChain =
      createBasicRedditSearchAnsweringChain(llm, embeddings, optimizationMode);
    const stream = basicRedditSearchAnsweringChain.streamEvents(
      {
        chat_history: history,
        query: query,
      },
      {
        version: 'v1',
      },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
      'error',
      JSON.stringify({ data: 'An error has occurred please try again later' }),
    );
    logger.error(`Error in RedditSearch: ${err}`);
  }

  return emitter;
};

const handleRedditSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = basicRedditSearch(
    message,
    history,
    llm,
    embeddings,
    optimizationMode,
  );
  return emitter;
};

export default handleRedditSearch;

<environment_details>
# VSCode Visible Files
backend/agents/image_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
After comparing the reddit search agents, I notice a few differences that should be synchronized:

1. The TypeScript version has additional details in the response prompt about citation numbers
2. The TypeScript version has a more sophisticated reranking process with a similarity threshold
3. The Python version's prompt formatting could be improved to match TypeScript's structure

Let me update the Python version to match these improvements.
</thinking>

<write_to_file>
<path>backend/agents/reddit_search_agent.py</path>
<content>"""
Reddit Search Agent Module

This module provides functionality for performing Reddit searches using the SearxNG search engine.
It includes document reranking capabilities to prioritize the most relevant Reddit discussions
and comments.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent
from utils.compute_similarity import compute_similarity

BASIC_REDDIT_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: Which company is most likely to create an AGI
Rephrased: Which company is most likely to create an AGI

2. Follow up question: Is Earth flat?
Rephrased: Is Earth flat?

3. Follow up question: Is there life on Mars?
Rephrased: Is there life on Mars?

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class RedditSearchAgent(AbstractAgent):
    """Agent for performing Reddit searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the Reddit search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_REDDIT_SEARCH_RETRIEVER_PROMPT)
        str_parser = StrOutputParser()

        chain = prompt | self.llm | str_parser

        async def process_output(query_text: str) -> Dict[str, Any]:
            """Process the LLM output and perform the Reddit search."""
            if query_text.strip() == "not_needed":
                return {"query": "", "docs": []}

            try:
                res = await search_searxng(query_text, {
                    "language": "en",
                    "engines": ["reddit"]
                })

                documents = [
                    Document(
                        page_content=result.get("content", result["title"]),
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            **({"img_src": result["img_src"]} if result.get("img_src") else {})
                        }
                    )
                    for result in res["results"]
                ]

                return {"query": query_text, "docs": documents}
            except Exception as e:
                logger.error("Error in Reddit search: %s", str(e))
                return {"query": query_text, "docs": []}

        retriever_chain = RunnableSequence([
            {
                "rephrased_query": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["rephrased_query"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the Reddit search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                    }

                reranked_docs = await self.rerank_docs(retriever_result["query"], retriever_result["docs"])
                context = await self.process_docs(reranked_docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({"query": input_data["query"]})

                return {
                    "type": "response",
                    "data": response.content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in reranked_docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while processing Reddit search results"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Reddit', this means you will be searching for information, opinions and discussions on the web using Reddit.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
        However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

        Anything inside the following `context` HTML block provided below is for your knowledge returned by Reddit and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to
        talk about the context in your response.

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from Reddit and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query."""
        if not docs:
            return docs

        docs_with_content = [doc for doc in docs if doc.page_content and len(doc.page_content) > 0]

        if self.optimization_mode == "speed":
            return docs_with_content[:15]
        elif self.optimization_mode == "balanced":
            doc_embeddings = await self.embeddings.embed_documents([doc.page_content for doc in docs_with_content])
            query_embedding = await self.embeddings.embed_query(query)

            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                sim = compute_similarity(query_embedding, doc_embedding)
                similarities.append({"index": i, "similarity": sim})

            # Filter docs with similarity > 0.3 and take top 15
            sorted_docs = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
            filtered_docs = [docs_with_content[s["index"]] for s in sorted_docs if s["similarity"] > 0.3][:15]
            return filtered_docs

        return docs

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_reddit_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a Reddit search query and generate streaming responses.

    Args:
        query: The Reddit search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = RedditSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
</content>
</write_to_file>

---

**User:**

[write_to_file for 'backend/agents/reddit_search_agent.py'] Result:
The content was successfully saved to backend/agents/reddit_search_agent.py.
<environment_details>
# VSCode Visible Files
backend/agents/reddit_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
backend/agents/reddit_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me check the remaining agents to ensure complete synchronization. Let me look at the Wolfram Alpha search agent next.
</thinking>

<read_file>
<path>backend/agents/wolfram_alpha_search_agent.py</path>
</read_file>

---

**User:**

[read_file for 'backend/agents/wolfram_alpha_search_agent.py'] Result:
"""
Wolfram Alpha Search Agent Module

This module provides functionality for performing Wolfram Alpha searches using the SearxNG search engine.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

BASIC_WOLFRAM_ALPHA_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: What is the atomic radius of S?
Rephrased: Atomic radius of S

2. Follow up question: What is linear algebra?
Rephrased: Linear algebra

3. Follow up question: What is the third law of thermodynamics?
Rephrased: Third law of thermodynamics

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class WolframAlphaSearchAgent(AbstractAgent):
    """Agent for performing Wolfram Alpha searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the Wolfram Alpha search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_WOLFRAM_ALPHA_SEARCH_RETRIEVER_PROMPT)
        str_parser = StrOutputParser()

        chain = prompt | self.llm | str_parser

        async def process_output(query_text: str) -> Dict[str, Any]:
            """Process the LLM output and perform the Wolfram Alpha search."""
            if query_text.strip() == "not_needed":
                return {"query": "", "docs": []}

            try:
                res = await search_searxng(query_text, {
                    "language": "en",
                    "engines": ["wolframalpha"]
                })

                documents = [
                    Document(
                        page_content=result["content"],
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            **({"img_src": result["img_src"]} if "img_src" in result else {})
                        }
                    )
                    for result in res["results"]
                ]

                return {"query": query_text, "docs": documents}
            except Exception as e:
                logger.error("Error in Wolfram Alpha search: %s", str(e))
                return {"query": query_text, "docs": []}

        retriever_chain = RunnableSequence([
            {
                "rephrased_query": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["rephrased_query"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the Wolfram Alpha search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                    }

                docs = retriever_result["docs"]
                context = await self.process_docs(docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                return {
                    "type": "response",
                    "data": response.content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while generating the response"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Wolfram Alpha', this means you will be searching for information on the web using Wolfram Alpha. It is a computational knowledge engine that can answer factual queries and perform computations.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from Wolfram Alpha and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_wolfram_alpha_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a Wolfram Alpha search query and generate streaming responses.

    Args:
        query: The Wolfram Alpha search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = WolframAlphaSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result

<environment_details>
# VSCode Visible Files
backend/agents/reddit_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
backend/agents/reddit_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me read the TypeScript version of the Wolfram Alpha search agent to compare them.
</thinking>

<read_file>
<path>src/agents/wolframAlphaSearchAgent.ts</path>
</read_file>

---

**User:**

[read_file for 'src/agents/wolframAlphaSearchAgent.ts'] Result:
import { BaseMessage } from '@langchain/core/messages';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import { searchSearxng } from '../lib/searxng';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import logger from '../utils/logger';
import { IterableReadableStream } from '@langchain/core/utils/stream';

const basicWolframAlphaSearchRetrieverPrompt = `
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return \`not_needed\` as the response.

Example:
1. Follow up question: What is the atomic radius of S?
Rephrased: Atomic radius of S

2. Follow up question: What is linear algebra?
Rephrased: Linear algebra

3. Follow up question: What is the third law of thermodynamics?
Rephrased: Third law of thermodynamics

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
`;

const basicWolframAlphaSearchResponsePrompt = `
    You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Wolfram Alpha', this means you will be searching for information on the web using Wolfram Alpha. It is a computational knowledge engine that can answer factual queries and perform computations.

    Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
    You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
    You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
    Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
    You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
    Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
    However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

    Anything inside the following \`context\` HTML block provided below is for your knowledge returned by Wolfram Alpha and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to 
    talk about the context in your response. 

    <context>
    {context}
    </context>

    If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
    Anything between the \`context\` is retrieved from Wolfram Alpha and is not a part of the conversation with the user. Today's date is ${new Date().toISOString()}
`;

const strParser = new StringOutputParser();

const handleStream = async (
  stream: IterableReadableStream<StreamEvent>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalSourceRetriever'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'sources', data: event.data.output }),
      );
    }
    if (
      event.event === 'on_chain_stream' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

type BasicChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const createBasicWolframAlphaSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicWolframAlphaSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'en',
        engines: ['wolframalpha'],
      });

      const documents = res.results.map(
        (result) =>
          new Document({
            pageContent: result.content,
            metadata: {
              title: result.title,
              url: result.url,
              ...(result.img_src && { img_src: result.img_src }),
            },
          }),
      );

      return { query: input, docs: documents };
    }),
  ]);
};

const createBasicWolframAlphaSearchAnsweringChain = (llm: BaseChatModel) => {
  const basicWolframAlphaSearchRetrieverChain =
    createBasicWolframAlphaSearchRetrieverChain(llm);

  const processDocs = (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
  };

  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: BasicChainInput) => input.query,
      chat_history: (input: BasicChainInput) => input.chat_history,
      context: RunnableSequence.from([
        (input) => ({
          query: input.query,
          chat_history: formatChatHistoryAsString(input.chat_history),
        }),
        basicWolframAlphaSearchRetrieverChain
          .pipe(({ query, docs }) => {
            return docs;
          })
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicWolframAlphaSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicWolframAlphaSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
) => {
  const emitter = new eventEmitter();

  try {
    const basicWolframAlphaSearchAnsweringChain =
      createBasicWolframAlphaSearchAnsweringChain(llm);
    const stream = basicWolframAlphaSearchAnsweringChain.streamEvents(
      {
        chat_history: history,
        query: query,
      },
      {
        version: 'v1',
      },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
      'error',
      JSON.stringify({ data: 'An error has occurred please try again later' }),
    );
    logger.error(`Error in WolframAlphaSearch: ${err}`);
  }

  return emitter;
};

const handleWolframAlphaSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = basicWolframAlphaSearch(message, history, llm);
  return emitter;
};

export default handleWolframAlphaSearch;

<environment_details>
# VSCode Visible Files
backend/agents/reddit_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
backend/agents/reddit_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
After comparing the Wolfram Alpha search agents, I notice a few differences that should be synchronized:

1. The TypeScript version has additional details in the response prompt about citation numbers
2. The TypeScript version doesn't use reranking for Wolfram Alpha results (which makes sense since Wolfram Alpha results are already highly relevant)
3. The Python version should be simplified to match TypeScript's approach

Let me update the Python version to match these improvements.
</thinking>

<write_to_file>
<path>backend/agents/wolfram_alpha_search_agent.py</path>
<content>"""
Wolfram Alpha Search Agent Module

This module provides functionality for performing Wolfram Alpha searches using the SearxNG search engine.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

BASIC_WOLFRAM_ALPHA_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: What is the atomic radius of S?
Rephrased: Atomic radius of S

2. Follow up question: What is linear algebra?
Rephrased: Linear algebra

3. Follow up question: What is the third law of thermodynamics?
Rephrased: Third law of thermodynamics

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class WolframAlphaSearchAgent(AbstractAgent):
    """Agent for performing Wolfram Alpha searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the Wolfram Alpha search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_WOLFRAM_ALPHA_SEARCH_RETRIEVER_PROMPT)
        str_parser = StrOutputParser()

        chain = prompt | self.llm | str_parser

        async def process_output(query_text: str) -> Dict[str, Any]:
            """Process the LLM output and perform the Wolfram Alpha search."""
            if query_text.strip() == "not_needed":
                return {"query": "", "docs": []}

            try:
                res = await search_searxng(query_text, {
                    "language": "en",
                    "engines": ["wolframalpha"]
                })

                documents = [
                    Document(
                        page_content=result["content"],
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            **({"img_src": result["img_src"]} if "img_src" in result else {})
                        }
                    )
                    for result in res["results"]
                ]

                return {"query": query_text, "docs": documents}
            except Exception as e:
                logger.error("Error in Wolfram Alpha search: %s", str(e))
                return {"query": query_text, "docs": []}

        retriever_chain = RunnableSequence([
            {
                "rephrased_query": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["rephrased_query"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the Wolfram Alpha search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                    }

                # No reranking needed for Wolfram Alpha results as they are already highly relevant
                docs = retriever_result["docs"]
                context = await self.process_docs(docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                return {
                    "type": "response",
                    "data": response.content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while generating the response"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Wolfram Alpha', this means you will be searching for information on the web using Wolfram Alpha. It is a computational knowledge engine that can answer factual queries and perform computations.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
        However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

        Anything inside the following `context` HTML block provided below is for your knowledge returned by Wolfram Alpha and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to 
        talk about the context in your response. 

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from Wolfram Alpha and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_wolfram_alpha_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a Wolfram Alpha search query and generate streaming responses.

    Args:
        query: The Wolfram Alpha search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model (unused for Wolfram Alpha search)
        optimization_mode: The optimization mode (unused for Wolfram Alpha search)

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = WolframAlphaSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
</content>
</write_to_file>

---

**User:**

[write_to_file for 'backend/agents/wolfram_alpha_search_agent.py'] Result:
The content was successfully saved to backend/agents/wolfram_alpha_search_agent.py.
<environment_details>
# VSCode Visible Files
backend/agents/wolfram_alpha_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
backend/agents/reddit_search_agent.py
backend/agents/wolfram_alpha_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me check the YouTube search agent next to ensure complete synchronization.
</thinking>

<read_file>
<path>backend/agents/youtube_search_agent.py</path>
</read_file>

---

**User:**

[read_file for 'backend/agents/youtube_search_agent.py'] Result:
"""
YouTube Search Agent Module

This module provides functionality for performing YouTube searches using the SearxNG search engine.
It includes document reranking capabilities for improved result relevance.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

BASIC_YOUTUBE_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: How does an A.C work?
Rephrased: A.C working

2. Follow up question: Linear algebra explanation video
Rephrased: What is linear algebra?

3. Follow up question: What is theory of relativity?
Rephrased: What is theory of relativity?

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class YouTubeSearchAgent(AbstractAgent):
    """Agent for performing YouTube searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the YouTube search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_YOUTUBE_SEARCH_RETRIEVER_PROMPT)
        str_parser = StrOutputParser()

        chain = prompt | self.llm | str_parser

        async def process_output(query_text: str) -> Dict[str, Any]:
            """Process the LLM output and perform the YouTube search."""
            if query_text.strip() == "not_needed":
                return {"query": "", "docs": []}

            try:
                res = await search_searxng(query_text, {
                    "language": "en",
                    "engines": ["youtube"]
                })

                documents = [
                    Document(
                        page_content=result.get("content", result["title"]),
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            **({"img_src": result["img_src"]} if "img_src" in result else {})
                        }
                    )
                    for result in res["results"]
                ]

                return {"query": query_text, "docs": documents}
            except Exception as e:
                logger.error("Error in YouTube search: %s", str(e))
                return {"query": query_text, "docs": []}

        retriever_chain = RunnableSequence([
            {
                "rephrased_query": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["rephrased_query"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the YouTube search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                    }

                reranked_docs = await self.rerank_docs(retriever_result["query"], retriever_result["docs"])
                context = await self.process_docs(reranked_docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                return {
                    "type": "response",
                    "data": response.content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in reranked_docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while generating the response"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Youtube', this means you will be searching for videos on the web using Youtube and providing information based on the video's transcript.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from Youtube and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_youtube_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a YouTube search query and generate streaming responses.

    Args:
        query: The YouTube search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = YouTubeSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result

<environment_details>
# VSCode Visible Files
backend/agents/wolfram_alpha_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
backend/agents/reddit_search_agent.py
backend/agents/wolfram_alpha_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
Let me read the TypeScript version of the YouTube search agent to compare them.
</thinking>

<read_file>
<path>src/agents/youtubeSearchAgent.ts</path>
</read_file>

---

**User:**

[read_file for 'src/agents/youtubeSearchAgent.ts'] Result:
import { BaseMessage } from '@langchain/core/messages';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import { searchSearxng } from '../lib/searxng';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import computeSimilarity from '../utils/computeSimilarity';
import logger from '../utils/logger';
import { IterableReadableStream } from '@langchain/core/utils/stream';

const basicYoutubeSearchRetrieverPrompt = `
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return \`not_needed\` as the response.

Example:
1. Follow up question: How does an A.C work?
Rephrased: A.C working

2. Follow up question: Linear algebra explanation video
Rephrased: What is linear algebra?

3. Follow up question: What is theory of relativity?
Rephrased: What is theory of relativity?

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
`;

const basicYoutubeSearchResponsePrompt = `
    You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Youtube', this means you will be searching for videos on the web using Youtube and providing information based on the video's transcript.

    Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
    You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
    You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
    Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
    You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
    Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
    However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

    Anything inside the following \`context\` HTML block provided below is for your knowledge returned by Youtube and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to
    talk about the context in your response.

    <context>
    {context}
    </context>

    If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
    Anything between the \`context\` is retrieved from Youtube and is not a part of the conversation with the user. Today's date is ${new Date().toISOString()}
`;

const strParser = new StringOutputParser();

const handleStream = async (
  stream: IterableReadableStream<StreamEvent>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalSourceRetriever'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'sources', data: event.data.output }),
      );
    }
    if (
      event.event === 'on_chain_stream' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

type BasicChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const createBasicYoutubeSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicYoutubeSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'en',
        engines: ['youtube'],
      });

      const documents = res.results.map(
        (result) =>
          new Document({
            pageContent: result.content ? result.content : result.title,
            metadata: {
              title: result.title,
              url: result.url,
              ...(result.img_src && { img_src: result.img_src }),
            },
          }),
      );

      return { query: input, docs: documents };
    }),
  ]);
};

const createBasicYoutubeSearchAnsweringChain = (
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const basicYoutubeSearchRetrieverChain =
    createBasicYoutubeSearchRetrieverChain(llm);

  const processDocs = async (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
  };

  const rerankDocs = async ({
    query,
    docs,
  }: {
    query: string;
    docs: Document[];
  }) => {
    if (docs.length === 0) {
      return docs;
    }

    const docsWithContent = docs.filter(
      (doc) => doc.pageContent && doc.pageContent.length > 0,
    );

    if (optimizationMode === 'speed') {
      return docsWithContent.slice(0, 15);
    } else {
      const [docEmbeddings, queryEmbedding] = await Promise.all([
        embeddings.embedDocuments(
          docsWithContent.map((doc) => doc.pageContent),
        ),
        embeddings.embedQuery(query),
      ]);

      const similarity = docEmbeddings.map((docEmbedding, i) => {
        const sim = computeSimilarity(queryEmbedding, docEmbedding);

        return {
          index: i,
          similarity: sim,
        };
      });

      const sortedDocs = similarity
        .filter((sim) => sim.similarity > 0.3)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 15)
        .map((sim) => docsWithContent[sim.index]);

      return sortedDocs;
    }
  };

  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: BasicChainInput) => input.query,
      chat_history: (input: BasicChainInput) => input.chat_history,
      context: RunnableSequence.from([
        (input) => ({
          query: input.query,
          chat_history: formatChatHistoryAsString(input.chat_history),
        }),
        basicYoutubeSearchRetrieverChain
          .pipe(rerankDocs)
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicYoutubeSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicYoutubeSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = new eventEmitter();

  try {
    const basicYoutubeSearchAnsweringChain =
      createBasicYoutubeSearchAnsweringChain(llm, embeddings, optimizationMode);

    const stream = basicYoutubeSearchAnsweringChain.streamEvents(
      {
        chat_history: history,
        query: query,
      },
      {
        version: 'v1',
      },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
      'error',
      JSON.stringify({ data: 'An error has occurred please try again later' }),
    );
    logger.error(`Error in youtube search: ${err}`);
  }

  return emitter;
};

const handleYoutubeSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = basicYoutubeSearch(
    message,
    history,
    llm,
    embeddings,
    optimizationMode,
  );
  return emitter;
};

export default handleYoutubeSearch;

<environment_details>
# VSCode Visible Files
backend/agents/wolfram_alpha_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
backend/agents/reddit_search_agent.py
backend/agents/wolfram_alpha_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
After comparing the YouTube search agents, I notice a few differences that should be synchronized:

1. The TypeScript version has additional details in the response prompt about citation numbers
2. The TypeScript version has a more sophisticated reranking process with a similarity threshold
3. The Python version's prompt formatting could be improved to match TypeScript's structure

Let me update the Python version to match these improvements.
</thinking>

<write_to_file>
<path>backend/agents/youtube_search_agent.py</path>
<content>"""
YouTube Search Agent Module

This module provides functionality for performing YouTube searches using the SearxNG search engine.
It includes document reranking capabilities for improved result relevance.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent
from utils.compute_similarity import compute_similarity

BASIC_YOUTUBE_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: How does an A.C work?
Rephrased: A.C working

2. Follow up question: Linear algebra explanation video
Rephrased: What is linear algebra?

3. Follow up question: What is theory of relativity?
Rephrased: What is theory of relativity?

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class YouTubeSearchAgent(AbstractAgent):
    """Agent for performing YouTube searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the YouTube search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_YOUTUBE_SEARCH_RETRIEVER_PROMPT)
        str_parser = StrOutputParser()

        chain = prompt | self.llm | str_parser

        async def process_output(query_text: str) -> Dict[str, Any]:
            """Process the LLM output and perform the YouTube search."""
            if query_text.strip() == "not_needed":
                return {"query": "", "docs": []}

            try:
                res = await search_searxng(query_text, {
                    "language": "en",
                    "engines": ["youtube"]
                })

                documents = [
                    Document(
                        page_content=result.get("content", result["title"]),
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            **({"img_src": result["img_src"]} if "img_src" in result else {})
                        }
                    )
                    for result in res["results"]
                ]

                return {"query": query_text, "docs": documents}
            except Exception as e:
                logger.error("Error in YouTube search: %s", str(e))
                return {"query": query_text, "docs": []}

        retriever_chain = RunnableSequence([
            {
                "rephrased_query": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["rephrased_query"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the YouTube search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                    }

                reranked_docs = await self.rerank_docs(retriever_result["query"], retriever_result["docs"])
                context = await self.process_docs(reranked_docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                return {
                    "type": "response",
                    "data": response.content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in reranked_docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while generating the response"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Youtube', this means you will be searching for videos on the web using Youtube and providing information based on the video's transcript.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
        However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

        Anything inside the following `context` HTML block provided below is for your knowledge returned by Youtube and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to
        talk about the context in your response.

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from Youtube and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query."""
        if not docs:
            return docs

        docs_with_content = [doc for doc in docs if doc.page_content and len(doc.page_content) > 0]

        if self.optimization_mode == "speed":
            return docs_with_content[:15]
        else:
            doc_embeddings = await self.embeddings.embed_documents([doc.page_content for doc in docs_with_content])
            query_embedding = await self.embeddings.embed_query(query)

            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                sim = compute_similarity(query_embedding, doc_embedding)
                similarities.append({"index": i, "similarity": sim})

            # Filter docs with similarity > 0.3 and take top 15
            sorted_docs = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
            filtered_docs = [docs_with_content[s["index"]] for s in sorted_docs if s["similarity"] > 0.3][:15]
            return filtered_docs

        return docs

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_youtube_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a YouTube search query and generate streaming responses.

    Args:
        query: The YouTube search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = YouTubeSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
</content>
</write_to_file>

---

**User:**

[write_to_file for 'backend/agents/youtube_search_agent.py'] Result:
The content was successfully saved to backend/agents/youtube_search_agent.py.
<environment_details>
# VSCode Visible Files
backend/agents/youtube_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
backend/agents/reddit_search_agent.py
backend/agents/wolfram_alpha_search_agent.py
backend/agents/youtube_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
After reviewing and updating all the agent pairs, here's a summary of the synchronization work completed:

1. Academic Search Agent:
   - Updated with more detailed citation instructions
   - Added explicit context handling instructions

2. Web Search Agent:
   - Added more detailed examples in retriever prompt
   - Added similarity threshold for reranking
   - Added explicit temperature setting

3. Image Search Agent:
   - Added query rephrasing step
   - Added specific image search engines
   - Limited results to 10 images

4. Reddit Search Agent:
   - Added citation number clarification
   - Added similarity threshold for reranking
   - Improved prompt formatting

5. Wolfram Alpha Search Agent:
   - Added citation number clarification
   - Removed unnecessary reranking
   - Improved prompt formatting

6. YouTube Search Agent:
   - Added citation number clarification
   - Added similarity threshold for reranking
   - Improved prompt formatting

All agents are now synchronized with their TypeScript counterparts while maintaining the Python codebase's architectural requirements (AbstractAgent class structure). The main improvements across all agents were:
1. More consistent prompt formatting
2. Better citation instructions
3. Improved reranking where applicable
4. Specific optimizations for each search type

<thinking>
Let me verify if there are any other agents that need to be checked.
</thinking>

<list_files>
<path>backend/agents</path>
</list_files>

---

**User:**

[list_files for 'backend/agents'] Result:
__init__.py
abstract_agent.py
academic_search_agent.py
image_search_agent.py
reddit_search_agent.py
suggestion_generator_agent.py
video_search_agent.py
web_search_agent.py
wolfram_alpha_search_agent.py
writing_assistant_agent.py
youtube_search_agent.py
__pycache__/
<environment_details>
# VSCode Visible Files
backend/agents/youtube_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
backend/agents/reddit_search_agent.py
backend/agents/wolfram_alpha_search_agent.py
backend/agents/youtube_search_agent.py
</environment_details>

---

**Assistant:**

<thinking>
I notice there's a video_search_agent.py in the Python codebase that we haven't checked yet. Let me verify if this is a duplicate of youtube_search_agent.py or if it serves a different purpose.
</thinking>

<read_file>
<path>backend/agents/video_search_agent.py</path>
</read_file>

---

**User:**

[read_file for 'backend/agents/video_search_agent.py'] Result:
"""
Video Search Agent Module

This module provides functionality for performing video searches using the SearxNG search engine
with YouTube as the primary source. It specifically handles video content with support for
thumbnails and embedded iframes.
"""

from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

VIDEO_SEARCH_CHAIN_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question so it is a standalone question that can be used by the LLM to search Youtube for videos.
You need to make sure the rephrased question agrees with the conversation and is relevant to the conversation.

Example:
1. Follow up question: How does a car work?
Rephrased: How does a car work?

2. Follow up question: What is the theory of relativity?
Rephrased: What is theory of relativity

3. Follow up question: How does an AC work?
Rephrased: How does an AC work

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class VideoSearchAgent(AbstractAgent):
    """Agent for performing video searches and retrieving results."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the video search retriever chain."""
        prompt = PromptTemplate.from_template(VIDEO_SEARCH_CHAIN_PROMPT)
        str_parser = StrOutputParser()

        chain = prompt | self.llm | str_parser

        async def perform_search(query: str) -> Dict[str, Any]:
            """Perform the video search using YouTube through SearxNG."""
            try:
                res = await search_searxng(query, {
                    "engines": ["youtube"]
                })

                documents = []
                for result in res["results"]:
                    if (result.get("thumbnail") and
                        result.get("url") and
                        result.get("title") and
                        result.get("iframe_src")):
                        documents.append(Document(
                            page_content="",  # Video searches don't need text content
                            metadata={
                                "img_src": result["thumbnail"],
                                "url": result["url"],
                                "title": result["title"],
                                "iframe_src": result["iframe_src"]
                            }
                        ))

                return {"query": query, "docs": documents[:10]}  # Return top 10 videos

            except Exception as e:
                logger.error("Error in video search: %s", str(e))
                return {"query": query, "docs": []}

        retriever_chain = RunnableSequence([
            {
                "rephrased_query": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: perform_search(x["rephrased_query"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the video search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def process_results(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the retriever results into the final format."""
            try:
                result = await retriever_chain.invoke(input_data)

                if not result["docs"]:
                    return {
                        "type": "response",
                        "data": []
                    }

                # Extract video data from documents
                videos = [
                    {
                        "img_src": doc.metadata["img_src"],
                        "url": doc.metadata["url"],
                        "title": doc.metadata["title"],
                        "iframe_src": doc.metadata["iframe_src"]
                    }
                    for doc in result["docs"]
                ]

                return {
                    "type": "response",
                    "data": videos
                }

            except Exception as e:
                logger.error("Error processing video search results: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while processing the video search results"
                }

        return RunnableSequence([process_results])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        # Not used for video search as it doesn't require text generation
        return ""

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        # Not used for video search as it doesn't require text generation
        return {
            "type": "response",
            "data": response
        }

async def handle_video_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> List[Dict[str, str]]:
    """
    Handle a video search query.

    Args:
        query: The video search query
        history: The chat history
        llm: The language model to use for query processing
        embeddings: The embeddings model (unused for video search)
        optimization_mode: The optimization mode (unused for video search)

    Returns:
        List[Dict[str, str]]: A list of video results, each containing:
            - img_src: URL of the video thumbnail
            - url: URL of the video page
            - title: Title of the video
            - iframe_src: Embedded iframe URL for the video
    """
    agent = VideoSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        if result["type"] == "response":
            return result["data"]
    return []  # Return empty list if no results or error

<environment_details>
# VSCode Visible Files
backend/agents/youtube_search_agent.py

# VSCode Open Tabs
backend/agents/academic_search_agent.py
backend/agents/web_search_agent.py
backend/agents/image_search_agent.py
backend/agents/reddit_search_agent.py
backend/agents/wolfram_alpha_search_agent.py
backend/agents/youtube_search_agent.py
</environment_details>

