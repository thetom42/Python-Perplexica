**User:**

<task>
Explain the code in the src subdir
</task>
<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)

# Current Working Directory (/home/the/src/repos/other/perplexica) Files
.dockerignore
.gitignore
.prettierignore
.prettierrc.js
app.dockerfile
backend.dockerfile
CONTRIBUTING.md
docker-compose.yaml
drizzle.config.ts
LICENSE
package.json
README.md
sample.config.toml
tsconfig.json
yarn.lock
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
src/lib/outputParsers/lineOutputParser.ts
src/lib/outputParsers/listLineOutputParser.ts
src/lib/providers/
src/lib/providers/anthropic.ts
src/lib/providers/groq.ts
src/lib/providers/index.ts
src/lib/providers/ollama.ts
src/lib/providers/openai.ts
src/lib/providers/transformers.ts
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
ui/app/c/[chatId]/
ui/app/c/[chatId]/page.tsx
ui/app/library/
ui/app/library/layout.tsx
ui/app/library/page.tsx
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
ui/components/SearchVideos.tsx
ui/components/SettingsDialog.tsx
ui/components/Sidebar.tsx
ui/components/MessageActions/
ui/components/MessageActions/Copy.tsx
ui/components/MessageActions/Rewrite.tsx
ui/components/MessageInputActions/
ui/components/MessageInputActions/Attach.tsx
ui/components/MessageInputActions/Copilot.tsx
ui/components/MessageInputActions/Focus.tsx
ui/components/theme/
ui/components/theme/Provider.tsx
ui/components/theme/Switcher.tsx
ui/lib/
ui/lib/actions.ts
ui/lib/utils.ts
ui/public/
ui/public/next.svg
ui/public/vercel.svg
</environment_details>

---

**Assistant:**

To explain the code in the src subdirectory, I'll need to examine the contents of the relevant files. Let's start by looking at the main application file and then explore the different components of the project.
<read_file>
<path>
src/app.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import { startWebSocketServer } from './websocket';
import express from 'express';
import cors from 'cors';
import http from 'http';
import routes from './routes';
import { getPort } from './config';
import logger from './utils/logger';

const port = getPort();

const app = express();
const server = http.createServer(app);

const corsOptions = {
  origin: '*',
};

app.use(cors(corsOptions));
app.use(express.json());

app.use('/api', routes);
app.get('/api', (_, res) => {
  res.status(200).json({ status: 'ok' });
});

server.listen(port, () => {
  logger.info(`Server is running on port ${port}`);
});

startWebSocketServer(server);

process.on('uncaughtException', (err, origin) => {
  logger.error(`Uncaught Exception at ${origin}: ${err}`);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error(`Unhandled Rejection at: ${promise}, reason: ${reason}`);
});

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/app.ts` file. I'll now explain the code structure in the src subdirectory, starting with this main application file and then exploring other key components.

1. Main Application (src/app.ts):
   This file sets up the main Express server and WebSocket server for the application.

   - It imports necessary modules and local dependencies.
   - Sets up an Express app with CORS enabled.
   - Configures routes under the '/api' path.
   - Starts the HTTP server on a specified port.
   - Initializes the WebSocket server.
   - Sets up global error handlers for uncaught exceptions and unhandled rejections.

2. Configuration (src/config.ts):
Let's examine this file to understand how configuration is managed.
<read_file>
<path>
src/config.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import fs from 'fs';
import path from 'path';
import toml from '@iarna/toml';

const configFileName = 'config.toml';

interface Config {
  GENERAL: {
    PORT: number;
    SIMILARITY_MEASURE: string;
  };
  API_KEYS: {
    OPENAI: string;
    GROQ: string;
    ANTHROPIC: string;
  };
  API_ENDPOINTS: {
    SEARXNG: string;
    OLLAMA: string;
  };
}

type RecursivePartial<T> = {
  [P in keyof T]?: RecursivePartial<T[P]>;
};

const loadConfig = () =>
  toml.parse(
    fs.readFileSync(path.join(__dirname, `../${configFileName}`), 'utf-8'),
  ) as any as Config;

export const getPort = () => loadConfig().GENERAL.PORT;

export const getSimilarityMeasure = () =>
  loadConfig().GENERAL.SIMILARITY_MEASURE;

export const getOpenaiApiKey = () => loadConfig().API_KEYS.OPENAI;

export const getGroqApiKey = () => loadConfig().API_KEYS.GROQ;

export const getAnthropicApiKey = () => loadConfig().API_KEYS.ANTHROPIC;

export const getSearxngApiEndpoint = () =>
  process.env.SEARXNG_API_URL || loadConfig().API_ENDPOINTS.SEARXNG;

export const getOllamaApiEndpoint = () => loadConfig().API_ENDPOINTS.OLLAMA;

export const updateConfig = (config: RecursivePartial<Config>) => {
  const currentConfig = loadConfig();

  for (const key in currentConfig) {
    if (!config[key]) config[key] = {};

    if (typeof currentConfig[key] === 'object' && currentConfig[key] !== null) {
      for (const nestedKey in currentConfig[key]) {
        if (
          !config[key][nestedKey] &&
          currentConfig[key][nestedKey] &&
          config[key][nestedKey] !== ''
        ) {
          config[key][nestedKey] = currentConfig[key][nestedKey];
        }
      }
    } else if (currentConfig[key] && config[key] !== '') {
      config[key] = currentConfig[key];
    }
  }

  fs.writeFileSync(
    path.join(__dirname, `../${configFileName}`),
    toml.stringify(config),
  );
};

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/config.ts` file. I'll continue explaining the code structure in the src subdirectory, focusing on the configuration management and other key components.

2. Configuration Management (src/config.ts):
   This file handles loading and managing the application's configuration.

   - It uses TOML format for configuration.
   - Defines a `Config` interface that outlines the structure of the configuration.
   - Provides functions to load the config and retrieve specific values (e.g., API keys, endpoints).
   - Includes a function to update the configuration file.

3. Routes:
   The application uses a modular routing structure. Let's examine the main routes file:
<read_file>
<path>
src/routes/index.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import express from 'express';
import imagesRouter from './images';
import videosRouter from './videos';
import configRouter from './config';
import modelsRouter from './models';
import suggestionsRouter from './suggestions';
import chatsRouter from './chats';
import searchRouter from './search';

const router = express.Router();

router.use('/images', imagesRouter);
router.use('/videos', videosRouter);
router.use('/config', configRouter);
router.use('/models', modelsRouter);
router.use('/suggestions', suggestionsRouter);
router.use('/chats', chatsRouter);
router.use('/search', searchRouter);

export default router;

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/routes/index.ts` file. I'll continue explaining the code structure in the src subdirectory, focusing on the routing system and other key components.

3. Routing System (src/routes/index.ts):
   This file sets up the main router for the application, organizing different functionalities into separate route modules.

   - It imports various route modules (images, videos, config, models, suggestions, chats, search).
   - Creates an Express router instance.
   - Mounts each imported router on a specific path (e.g., '/images', '/videos', etc.).

4. Agents:
   The `src/agents` directory contains various agent modules for different functionalities. Let's examine one of these agents:
<read_file>
<path>
src/agents/webSearchAgent.ts
</path>
</read_file>

---

**User:**

[read_file Result]

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
            You are a text summarizer. You need to summarize the text provided inside the \`text\` XML block. 
            You need to summarize the text into 1 or 2 sentences capturing the main idea of the text.
            You need to make sure that you don't miss any point while summarizing the text.
            You will also be given a \`query\` XML block which will contain the query of the user. Try to answer the query in the summary from the text provided.
            If the query says Summarize then you just need to summarize the text without answering the query.
            Only return the summarized text without any other messages, text or XML block.

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

    const [docEmbeddings, queryEmbedding] = await Promise.all([
      embeddings.embedDocuments(docsWithContent.map((doc) => doc.pageContent)),
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
) => {
  const emitter = new eventEmitter();

  try {
    const basicWebSearchAnsweringChain = createBasicWebSearchAnsweringChain(
      llm,
      embeddings,
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
) => {
  const emitter = basicWebSearch(message, history, llm, embeddings);
  return emitter;
};

export default handleWebSearch;

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/agents/webSearchAgent.ts` file. I'll continue explaining the code structure in the src subdirectory, focusing on this agent and other key components.

4. Web Search Agent (src/agents/webSearchAgent.ts):
   This file implements a sophisticated web search agent that can rephrase questions, perform web searches, and generate responses based on the search results.

   Key features:
   - Uses LangChain for creating a chain of operations (RunnableSequence).
   - Implements question rephrasing to create standalone search queries.
   - Performs web searches using SearxNG (a meta search engine).
   - Summarizes and processes search results.
   - Generates responses using a language model, citing sources.
   - Handles streaming of results and sources back to the client.

5. Database (src/db):
   Let's examine the database setup:
<read_file>
<path>
src/db/index.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import { drizzle } from 'drizzle-orm/better-sqlite3';
import Database from 'better-sqlite3';
import * as schema from './schema';

const sqlite = new Database('data/db.sqlite');
const db = drizzle(sqlite, {
  schema: schema,
});

export default db;

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/db/index.ts` file. I'll continue explaining the code structure in the src subdirectory, focusing on the database setup and other key components.

5. Database Setup (src/db/index.ts):
   This file sets up the database connection using Drizzle ORM with SQLite.

   - It uses `better-sqlite3` as the SQLite driver.
   - Creates a database connection to `data/db.sqlite`.
   - Initializes Drizzle ORM with the schema imported from `./schema`.

6. Database Schema:
   Let's examine the database schema:
<read_file>
<path>
src/db/schema.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import { text, integer, sqliteTable } from 'drizzle-orm/sqlite-core';

export const messages = sqliteTable('messages', {
  id: integer('id').primaryKey(),
  content: text('content').notNull(),
  chatId: text('chatId').notNull(),
  messageId: text('messageId').notNull(),
  role: text('type', { enum: ['assistant', 'user'] }),
  metadata: text('metadata', {
    mode: 'json',
  }),
});

export const chats = sqliteTable('chats', {
  id: text('id').primaryKey(),
  title: text('title').notNull(),
  createdAt: text('createdAt').notNull(),
  focusMode: text('focusMode').notNull(),
});

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/db/schema.ts` file. I'll continue explaining the code structure in the src subdirectory, focusing on the database schema and other key components.

6. Database Schema (src/db/schema.ts):
   This file defines the database schema using Drizzle ORM.

   - It defines two tables: `messages` and `chats`.
   - The `messages` table stores individual messages with fields for content, chat ID, message ID, role, and metadata.
   - The `chats` table stores chat sessions with fields for ID, title, creation time, and focus mode.

7. WebSocket Server:
   The application uses WebSockets for real-time communication. Let's examine the WebSocket server setup:
<read_file>
<path>
src/websocket/index.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import { initServer } from './websocketServer';
import http from 'http';

export const startWebSocketServer = (
  server: http.Server<typeof http.IncomingMessage, typeof http.ServerResponse>,
) => {
  initServer(server);
};

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/websocket/index.ts` file. I'll continue explaining the code structure in the src subdirectory, focusing on the WebSocket server setup and other key components.

7. WebSocket Server (src/websocket/index.ts):
   This file exports a function to start the WebSocket server.

   - It imports the `initServer` function from `./websocketServer`.
   - Exports a `startWebSocketServer` function that takes an HTTP server as an argument and initializes the WebSocket server.

Let's examine the actual WebSocket server implementation:
<read_file>
<path>
src/websocket/websocketServer.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import { WebSocketServer } from 'ws';
import { handleConnection } from './connectionManager';
import http from 'http';
import { getPort } from '../config';
import logger from '../utils/logger';

export const initServer = (
  server: http.Server<typeof http.IncomingMessage, typeof http.ServerResponse>,
) => {
  const port = getPort();
  const wss = new WebSocketServer({ server });

  wss.on('connection', handleConnection);

  logger.info(`WebSocket server started on port ${port}`);
};

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/websocket/websocketServer.ts` file. I'll continue explaining the code structure in the src subdirectory, focusing on the WebSocket server implementation and other key components.

8. WebSocket Server Implementation (src/websocket/websocketServer.ts):
   This file sets up the WebSocket server using the `ws` library.

   - It imports the `handleConnection` function from `./connectionManager`.
   - Creates a new WebSocketServer instance attached to the provided HTTP server.
   - Sets up a connection event listener that calls `handleConnection` for each new connection.
   - Logs a message when the server starts.

9. Connection Manager:
   Let's examine the connection manager to understand how WebSocket connections are handled:
<read_file>
<path>
src/websocket/connectionManager.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import { WebSocket } from 'ws';
import { handleMessage } from './messageHandler';
import {
  getAvailableEmbeddingModelProviders,
  getAvailableChatModelProviders,
} from '../lib/providers';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import type { IncomingMessage } from 'http';
import logger from '../utils/logger';
import { ChatOpenAI } from '@langchain/openai';

export const handleConnection = async (
  ws: WebSocket,
  request: IncomingMessage,
) => {
  try {
    const searchParams = new URL(request.url, `http://${request.headers.host}`)
      .searchParams;

    const [chatModelProviders, embeddingModelProviders] = await Promise.all([
      getAvailableChatModelProviders(),
      getAvailableEmbeddingModelProviders(),
    ]);

    const chatModelProvider =
      searchParams.get('chatModelProvider') ||
      Object.keys(chatModelProviders)[0];
    const chatModel =
      searchParams.get('chatModel') ||
      Object.keys(chatModelProviders[chatModelProvider])[0];

    const embeddingModelProvider =
      searchParams.get('embeddingModelProvider') ||
      Object.keys(embeddingModelProviders)[0];
    const embeddingModel =
      searchParams.get('embeddingModel') ||
      Object.keys(embeddingModelProviders[embeddingModelProvider])[0];

    let llm: BaseChatModel | undefined;
    let embeddings: Embeddings | undefined;

    if (
      chatModelProviders[chatModelProvider] &&
      chatModelProviders[chatModelProvider][chatModel] &&
      chatModelProvider != 'custom_openai'
    ) {
      llm = chatModelProviders[chatModelProvider][chatModel]
        .model as unknown as BaseChatModel | undefined;
    } else if (chatModelProvider == 'custom_openai') {
      llm = new ChatOpenAI({
        modelName: chatModel,
        openAIApiKey: searchParams.get('openAIApiKey'),
        temperature: 0.7,
        configuration: {
          baseURL: searchParams.get('openAIBaseURL'),
        },
      }) as unknown as BaseChatModel;
    }

    if (
      embeddingModelProviders[embeddingModelProvider] &&
      embeddingModelProviders[embeddingModelProvider][embeddingModel]
    ) {
      embeddings = embeddingModelProviders[embeddingModelProvider][
        embeddingModel
      ].model as Embeddings | undefined;
    }

    if (!llm || !embeddings) {
      ws.send(
        JSON.stringify({
          type: 'error',
          data: 'Invalid LLM or embeddings model selected, please refresh the page and try again.',
          key: 'INVALID_MODEL_SELECTED',
        }),
      );
      ws.close();
    }

    ws.on(
      'message',
      async (message) =>
        await handleMessage(message.toString(), ws, llm, embeddings),
    );

    ws.on('close', () => logger.debug('Connection closed'));
  } catch (err) {
    ws.send(
      JSON.stringify({
        type: 'error',
        data: 'Internal server error.',
        key: 'INTERNAL_SERVER_ERROR',
      }),
    );
    ws.close();
    logger.error(err);
  }
};

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/websocket/connectionManager.ts` file. I'll continue explaining the code structure in the src subdirectory, focusing on the connection manager and summarizing the overall architecture.

9. Connection Manager (src/websocket/connectionManager.ts):
   This file handles new WebSocket connections and sets up message handling.

   Key features:
   - Parses URL parameters to determine chat and embedding model providers and models.
   - Initializes language models (LLMs) and embeddings based on the selected providers.
   - Sets up message handling for each connection.
   - Handles errors and closes connections when necessary.

10. Overall Architecture Summary:

    The src directory contains a well-structured Node.js application with the following key components:

    a. Main Application (app.ts):
       - Sets up the Express server and WebSocket server.
       - Configures middleware, routes, and error handling.

    b. Configuration (config.ts):
       - Manages application configuration using TOML files.
       - Provides functions to access various configuration values.

    c. Routing (routes/):
       - Organizes API endpoints into separate route files (images, videos, config, models, suggestions, chats, search).

    d. Agents (agents/):
       - Implements various AI agents for different functionalities (e.g., webSearchAgent).
       - Uses LangChain for creating chains of operations.

    e. Database (db/):
       - Sets up a SQLite database using Drizzle ORM.
       - Defines schema for messages and chats tables.

    f. WebSocket Server (websocket/):
       - Implements real-time communication using WebSockets.
       - Manages connections and handles messages.

    g. Libraries (lib/):
       - Contains utility functions and integrations (e.g., SearxNG for web search).

    h. Utils (utils/):
       - Provides utility functions for various tasks (e.g., logging, similarity computation).

This architecture supports a flexible and modular AI-powered chat application with web search capabilities, real-time communication, and persistent storage. The use of modern libraries and frameworks (Express, WebSocket, LangChain, Drizzle ORM) allows for efficient development and scalability.

To complete the explanation of the src directory structure, I'll provide an overview of the lib and utils directories:
<list_code_definition_names>
<path>
src/lib
</path>
</list_code_definition_names>

---

**User:**

[list_code_definition_names Result]

huggingfaceTransformer.ts
|----
│export class HuggingFaceTransformersEmbeddings
|----
│  constructor(fields?: Partial<HuggingFaceTransformersEmbeddingsParams>) {
|----
│  async embedDocuments(texts: string[]): Promise<number[][]> {
|----
│  async embedQuery(text: string): Promise<number[]> {
|----
│  private async runEmbedding(texts: string[]) {
|----


<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the code definition names for the `src/lib` directory. I'll continue explaining the code structure in the src subdirectory, focusing on the lib and utils directories.

11. Libraries (src/lib):
    The lib directory contains various utility classes and functions that support the main application functionality. Let's examine some key components:

    a. HuggingFace Transformer (huggingfaceTransformer.ts):
       - Implements the `HuggingFaceTransformersEmbeddings` class for text embeddings.
       - Provides methods for embedding documents and queries using HuggingFace's transformer models.

    Let's look at other files in the lib directory:
<list_files>
<path>
src/lib
</path>
</list_files>

---

**User:**

[list_files Result]

huggingfaceTransformer.ts
linkDocument.ts
searxng.ts
outputParsers/
providers/
<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the list of files in the `src/lib` directory. I'll continue explaining the code structure, focusing on the remaining files and subdirectories in the lib directory.

11. Libraries (src/lib) continued:

    b. Link Document (linkDocument.ts):
       Let's examine this file:
<read_file>
<path>
src/lib/linkDocument.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import axios from 'axios';
import { htmlToText } from 'html-to-text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from '@langchain/core/documents';
import pdfParse from 'pdf-parse';
import logger from '../utils/logger';

export const getDocumentsFromLinks = async ({ links }: { links: string[] }) => {
  const splitter = new RecursiveCharacterTextSplitter();

  let docs: Document[] = [];

  await Promise.all(
    links.map(async (link) => {
      link =
        link.startsWith('http://') || link.startsWith('https://')
          ? link
          : `https://${link}`;

      try {
        const res = await axios.get(link, {
          responseType: 'arraybuffer',
        });

        const isPdf = res.headers['content-type'] === 'application/pdf';

        if (isPdf) {
          const pdfText = await pdfParse(res.data);
          const parsedText = pdfText.text
            .replace(/(\r\n|\n|\r)/gm, ' ')
            .replace(/\s+/g, ' ')
            .trim();

          const splittedText = await splitter.splitText(parsedText);
          const title = 'PDF Document';

          const linkDocs = splittedText.map((text) => {
            return new Document({
              pageContent: text,
              metadata: {
                title: title,
                url: link,
              },
            });
          });

          docs.push(...linkDocs);
          return;
        }

        const parsedText = htmlToText(res.data.toString('utf8'), {
          selectors: [
            {
              selector: 'a',
              options: {
                ignoreHref: true,
              },
            },
          ],
        })
          .replace(/(\r\n|\n|\r)/gm, ' ')
          .replace(/\s+/g, ' ')
          .trim();

        const splittedText = await splitter.splitText(parsedText);
        const title = res.data
          .toString('utf8')
          .match(/<title>(.*?)<\/title>/)?.[1];

        const linkDocs = splittedText.map((text) => {
          return new Document({
            pageContent: text,
            metadata: {
              title: title || link,
              url: link,
            },
          });
        });

        docs.push(...linkDocs);
      } catch (err) {
        logger.error(
          `Error at generating documents from links: ${err.message}`,
        );
        docs.push(
          new Document({
            pageContent: `Failed to retrieve content from the link: ${err.message}`,
            metadata: {
              title: 'Failed to retrieve content',
              url: link,
            },
          }),
        );
      }
    }),
  );

  return docs;
};

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/lib/linkDocument.ts` file. I'll continue explaining the code structure, focusing on this file and the remaining components in the lib directory.

11. Libraries (src/lib) continued:

    b. Link Document (linkDocument.ts):
       This file provides functionality to fetch and process documents from web links.

       Key features:
       - Fetches content from provided URLs using axios.
       - Handles both HTML and PDF content.
       - Converts HTML to plain text using html-to-text library.
       - Parses PDF content using pdf-parse library.
       - Splits text into smaller chunks using RecursiveCharacterTextSplitter.
       - Creates Document objects with page content and metadata (title, URL).
       - Handles errors and creates error documents when content retrieval fails.

    c. SearxNG (searxng.ts):
       Let's examine this file:
<read_file>
<path>
src/lib/searxng.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import axios from 'axios';
import { getSearxngApiEndpoint } from '../config';

interface SearxngSearchOptions {
  categories?: string[];
  engines?: string[];
  language?: string;
  pageno?: number;
}

interface SearxngSearchResult {
  title: string;
  url: string;
  img_src?: string;
  thumbnail_src?: string;
  thumbnail?: string;
  content?: string;
  author?: string;
  iframe_src?: string;
}

export const searchSearxng = async (
  query: string,
  opts?: SearxngSearchOptions,
) => {
  const searxngURL = getSearxngApiEndpoint();

  const url = new URL(`${searxngURL}/search?format=json`);
  url.searchParams.append('q', query);

  if (opts) {
    Object.keys(opts).forEach((key) => {
      if (Array.isArray(opts[key])) {
        url.searchParams.append(key, opts[key].join(','));
        return;
      }
      url.searchParams.append(key, opts[key]);
    });
  }

  const res = await axios.get(url.toString());

  const results: SearxngSearchResult[] = res.data.results;
  const suggestions: string[] = res.data.suggestions;

  return { results, suggestions };
};

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/lib/searxng.ts` file. I'll continue explaining the code structure, focusing on this file and the remaining components in the lib directory.

11. Libraries (src/lib) continued:

    c. SearxNG (searxng.ts):
       This file provides a function to perform web searches using the SearxNG meta search engine.

       Key features:
       - Defines interfaces for search options and results.
       - Implements a `searchSearxng` function that:
         - Constructs the search URL with query parameters.
         - Performs an HTTP GET request to the SearxNG API.
         - Returns search results and suggestions.

    d. Output Parsers (outputParsers/):
       This subdirectory likely contains custom output parsers for processing various types of data. Let's examine its contents:
<list_files>
<path>
src/lib/outputParsers
</path>
</list_files>

---

**User:**

[list_files Result]

lineOutputParser.ts
listLineOutputParser.ts
<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the list of files in the `src/lib/outputParsers` directory. I'll continue explaining the code structure, focusing on these output parsers and then move on to the providers subdirectory.

11. Libraries (src/lib) continued:

    d. Output Parsers (outputParsers/):
       This subdirectory contains custom output parsers for processing specific types of data.

       Let's examine the contents of these files:
<read_file>
<path>
src/lib/outputParsers/lineOutputParser.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import { BaseOutputParser } from '@langchain/core/output_parsers';

interface LineOutputParserArgs {
  key?: string;
}

class LineOutputParser extends BaseOutputParser<string> {
  private key = 'questions';

  constructor(args?: LineOutputParserArgs) {
    super();
    this.key = args.key ?? this.key;
  }

  static lc_name() {
    return 'LineOutputParser';
  }

  lc_namespace = ['langchain', 'output_parsers', 'line_output_parser'];

  async parse(text: string): Promise<string> {
    const regex = /^(\s*(-|\*|\d+\.\s|\d+\)\s|\u2022)\s*)+/;
    const startKeyIndex = text.indexOf(`<${this.key}>`);
    const endKeyIndex = text.indexOf(`</${this.key}>`);

    if (startKeyIndex === -1 || endKeyIndex === -1) {
      return '';
    }

    const questionsStartIndex =
      startKeyIndex === -1 ? 0 : startKeyIndex + `<${this.key}>`.length;
    const questionsEndIndex = endKeyIndex === -1 ? text.length : endKeyIndex;
    const line = text
      .slice(questionsStartIndex, questionsEndIndex)
      .trim()
      .replace(regex, '');

    return line;
  }

  getFormatInstructions(): string {
    throw new Error('Not implemented.');
  }
}

export default LineOutputParser;

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/lib/outputParsers/lineOutputParser.ts` file. I'll explain this parser and then move on to the `listLineOutputParser.ts` file.

11. Libraries (src/lib) continued:

    d. Output Parsers (outputParsers/):

       i. Line Output Parser (lineOutputParser.ts):
          This parser is designed to extract a single line of text from a structured output.

          Key features:
          - Extends BaseOutputParser from LangChain.
          - Searches for content between XML-like tags (e.g., <questions></questions>).
          - Removes leading bullet points, numbers, or other list markers.
          - Returns a single line of text.

       Let's examine the List Line Output Parser:
<read_file>
<path>
src/lib/outputParsers/listLineOutputParser.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import { BaseOutputParser } from '@langchain/core/output_parsers';

interface LineListOutputParserArgs {
  key?: string;
}

class LineListOutputParser extends BaseOutputParser<string[]> {
  private key = 'questions';

  constructor(args?: LineListOutputParserArgs) {
    super();
    this.key = args.key ?? this.key;
  }

  static lc_name() {
    return 'LineListOutputParser';
  }

  lc_namespace = ['langchain', 'output_parsers', 'line_list_output_parser'];

  async parse(text: string): Promise<string[]> {
    const regex = /^(\s*(-|\*|\d+\.\s|\d+\)\s|\u2022)\s*)+/;
    const startKeyIndex = text.indexOf(`<${this.key}>`);
    const endKeyIndex = text.indexOf(`</${this.key}>`);

    if (startKeyIndex === -1 || endKeyIndex === -1) {
      return [];
    }

    const questionsStartIndex =
      startKeyIndex === -1 ? 0 : startKeyIndex + `<${this.key}>`.length;
    const questionsEndIndex = endKeyIndex === -1 ? text.length : endKeyIndex;
    const lines = text
      .slice(questionsStartIndex, questionsEndIndex)
      .trim()
      .split('\n')
      .filter((line) => line.trim() !== '')
      .map((line) => line.replace(regex, ''));

    return lines;
  }

  getFormatInstructions(): string {
    throw new Error('Not implemented.');
  }
}

export default LineListOutputParser;

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/lib/outputParsers/listLineOutputParser.ts` file. I'll explain this parser and then move on to the providers subdirectory.

11. Libraries (src/lib) continued:

    d. Output Parsers (outputParsers/):

       ii. List Line Output Parser (listLineOutputParser.ts):
           This parser is designed to extract multiple lines of text from a structured output and return them as an array.

           Key features:
           - Extends BaseOutputParser from LangChain.
           - Searches for content between XML-like tags (e.g., <questions></questions>).
           - Splits the content into lines and filters out empty lines.
           - Removes leading bullet points, numbers, or other list markers from each line.
           - Returns an array of strings, each representing a line of text.

    Both parsers are useful for extracting structured data from the output of language models, making it easier to process and use the results in the application.

    e. Providers (providers/):
       This subdirectory likely contains implementations or configurations for various AI model providers. Let's examine its contents:
<list_files>
<path>
src/lib/providers
</path>
</list_files>

---

**User:**

[list_files Result]

anthropic.ts
groq.ts
index.ts
ollama.ts
openai.ts
transformers.ts
<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the list of files in the `src/lib/providers` directory. I'll explain the purpose of this directory and briefly describe each file:

11. Libraries (src/lib) continued:

    e. Providers (providers/):
       This subdirectory contains implementations or configurations for various AI model providers. Each file likely sets up the necessary clients or interfaces to interact with different AI services.

       i. anthropic.ts: Likely contains the setup for Anthropic's AI models (e.g., Claude).

       ii. groq.ts: Probably sets up the integration with Groq's AI services.

       iii. index.ts: Likely serves as the main entry point for the providers, possibly exporting all provider configurations.

       iv. ollama.ts: Probably contains the setup for Ollama, an open-source AI model serving platform.

       v. openai.ts: Likely sets up the integration with OpenAI's models (e.g., GPT-3.5, GPT-4).

       vi. transformers.ts: Probably contains configurations for using Hugging Face's Transformers library.

Let's examine the index.ts file to get an overview of how these providers are organized:
<read_file>
<path>
src/lib/providers/index.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import { loadGroqChatModels } from './groq';
import { loadOllamaChatModels, loadOllamaEmbeddingsModels } from './ollama';
import { loadOpenAIChatModels, loadOpenAIEmbeddingsModels } from './openai';
import { loadAnthropicChatModels } from './anthropic';
import { loadTransformersEmbeddingsModels } from './transformers';

const chatModelProviders = {
  openai: loadOpenAIChatModels,
  groq: loadGroqChatModels,
  ollama: loadOllamaChatModels,
  anthropic: loadAnthropicChatModels,
};

const embeddingModelProviders = {
  openai: loadOpenAIEmbeddingsModels,
  local: loadTransformersEmbeddingsModels,
  ollama: loadOllamaEmbeddingsModels,
};

export const getAvailableChatModelProviders = async () => {
  const models = {};

  for (const provider in chatModelProviders) {
    const providerModels = await chatModelProviders[provider]();
    if (Object.keys(providerModels).length > 0) {
      models[provider] = providerModels;
    }
  }

  models['custom_openai'] = {};

  return models;
};

export const getAvailableEmbeddingModelProviders = async () => {
  const models = {};

  for (const provider in embeddingModelProviders) {
    const providerModels = await embeddingModelProviders[provider]();
    if (Object.keys(providerModels).length > 0) {
      models[provider] = providerModels;
    }
  }

  return models;
};

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/lib/providers/index.ts` file. I'll explain this file and summarize the providers setup:

11. Libraries (src/lib) continued:

    e. Providers (providers/index.ts):
       This file serves as the main entry point for managing AI model providers. It organizes and exports functions to load available models from different providers.

       Key features:
       - Imports model loading functions from individual provider files.
       - Defines two main categories of models: chat models and embedding models.
       - Provides functions to get available chat and embedding model providers:
         - `getAvailableChatModelProviders()`: Loads and returns available chat models from OpenAI, Groq, Ollama, and Anthropic.
         - `getAvailableEmbeddingModelProviders()`: Loads and returns available embedding models from OpenAI, local transformers, and Ollama.
       - Includes a 'custom_openai' option for chat models, likely allowing users to specify custom OpenAI-compatible endpoints.

    Summary of Providers:
    1. Chat Model Providers:
       - OpenAI
       - Groq
       - Ollama
       - Anthropic
       - Custom OpenAI (for user-specified endpoints)

    2. Embedding Model Providers:
       - OpenAI
       - Local (using Transformers library)
       - Ollama

This setup allows the application to flexibly use different AI model providers for various tasks, such as chat completion and text embedding. The modular structure makes it easy to add or remove providers as needed.

To complete our overview of the src directory, let's examine the utils directory:
<list_files>
<path>
src/utils
</path>
</list_files>

---

**User:**

[list_files Result]

computeSimilarity.ts
formatHistory.ts
logger.ts
<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the list of files in the `src/utils` directory. I'll explain the purpose of this directory and briefly describe each file:

12. Utilities (src/utils):
    This directory contains utility functions that are used across the application. Let's examine each file:

    a. computeSimilarity.ts:
       Let's look at its content:
<read_file>
<path>
src/utils/computeSimilarity.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import dot from 'compute-dot';
import cosineSimilarity from 'compute-cosine-similarity';
import { getSimilarityMeasure } from '../config';

const computeSimilarity = (x: number[], y: number[]): number => {
  const similarityMeasure = getSimilarityMeasure();

  if (similarityMeasure === 'cosine') {
    return cosineSimilarity(x, y);
  } else if (similarityMeasure === 'dot') {
    return dot(x, y);
  }

  throw new Error('Invalid similarity measure');
};

export default computeSimilarity;

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/utils/computeSimilarity.ts` file. I'll explain this utility and then examine the remaining files in the utils directory.

12. Utilities (src/utils) continued:

    a. computeSimilarity.ts:
       This file provides a function to compute similarity between two vectors.

       Key features:
       - Uses the `compute-dot` and `compute-cosine-similarity` libraries for calculations.
       - Supports two similarity measures: cosine similarity and dot product.
       - The similarity measure is determined by the application configuration (retrieved using `getSimilarityMeasure()`).
       - Throws an error if an invalid similarity measure is specified.

    Let's examine the formatHistory.ts file:
<read_file>
<path>
src/utils/formatHistory.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import { BaseMessage } from '@langchain/core/messages';

const formatChatHistoryAsString = (history: BaseMessage[]) => {
  return history
    .map((message) => `${message._getType()}: ${message.content}`)
    .join('\n');
};

export default formatChatHistoryAsString;

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/utils/formatHistory.ts` file. I'll explain this utility and then examine the last file in the utils directory.

12. Utilities (src/utils) continued:

    b. formatHistory.ts:
       This file provides a function to format chat history into a string representation.

       Key features:
       - Takes an array of BaseMessage objects (likely from LangChain) as input.
       - Formats each message as a string with the format "message_type: message_content".
       - Joins all formatted messages with newline characters.
       - This formatted history can be used as context for AI models or for display purposes.

    Now, let's examine the logger.ts file:
<read_file>
<path>
src/utils/logger.ts
</path>
</read_file>

---

**User:**

[read_file Result]

import winston from 'winston';

const logger = winston.createLogger({
  level: 'info',
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple(),
      ),
    }),
    new winston.transports.File({
      filename: 'app.log',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json(),
      ),
    }),
  ],
});

export default logger;

<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

Thank you for providing the content of the `src/utils/logger.ts` file. I'll explain this utility and then summarize the overall structure of the src directory.

12. Utilities (src/utils) continued:

    c. logger.ts:
       This file sets up a logging system using the Winston library.

       Key features:
       - Creates a Winston logger instance.
       - Configures two transports:
         1. Console transport: 
            - Logs to the console with colorized output.
            - Uses a simple format for easy readability.
         2. File transport:
            - Logs to a file named 'app.log'.
            - Uses JSON format with timestamps for structured logging.
       - The logger is set to the 'info' level, which means it will log messages of level 'info' and above (warn, error).
       - This logger can be imported and used throughout the application for consistent logging.

Summary of the src directory structure:

1. Main Application (app.ts):
   - Sets up the Express server and WebSocket server.

2. Configuration (config.ts):
   - Manages application configuration.

3. Routing (routes/):
   - Organizes API endpoints.

4. Agents (agents/):
   - Implements AI agents for various functionalities.

5. Database (db/):
   - Sets up SQLite database using Drizzle ORM.

6. WebSocket Server (websocket/):
   - Implements real-time communication.

7. Libraries (lib/):
   - huggingfaceTransformer.ts: Implements text embeddings using HuggingFace models.
   - linkDocument.ts: Fetches and processes documents from web links.
   - searxng.ts: Performs web searches using SearxNG.
   - outputParsers/: Custom parsers for processing AI model outputs.
   - providers/: Configurations for various AI model providers.

8. Utilities (utils/):
   - computeSimilarity.ts: Calculates similarity between vectors.
   - formatHistory.ts: Formats chat history into strings.
   - logger.ts: Sets up application-wide logging.

This structure supports a flexible and modular AI-powered chat application with web search capabilities, real-time communication, and persistent storage. The use of modern libraries and frameworks (Express, WebSocket, LangChain, Drizzle ORM, Winston) allows for efficient development, scalability, and maintainability.

---

**User:**

If you have completed the user's task, use the attempt_completion tool. If you require additional information from the user, use the ask_followup_question tool. Otherwise, if you have not completed the task and do not need additional information, then proceed with the next step of the task. (This is an automated message, so do not respond to it conversationally.)
<environment_details>
# VSCode Visible Files
(No visible files)

# VSCode Open Tabs
(No open tabs)
</environment_details>

---

**Assistant:**

<attempt_completion>
<result>
I have completed the task of explaining the code in the src subdirectory. Here's a summary of the key components:

1. Main Application (app.ts): Sets up the Express and WebSocket servers.
2. Configuration (config.ts): Manages application settings using TOML files.
3. Routing (routes/): Organizes API endpoints for various functionalities.
4. Agents (agents/): Implements AI agents, including a sophisticated web search agent.
5. Database (db/): Sets up a SQLite database using Drizzle ORM for storing messages and chats.
6. WebSocket Server (websocket/): Handles real-time communication and manages connections.
7. Libraries (lib/):
   - huggingfaceTransformer.ts: Implements text embeddings using HuggingFace models.
   - linkDocument.ts: Fetches and processes documents from web links.
   - searxng.ts: Performs web searches using SearxNG.
   - outputParsers/: Custom parsers for processing AI model outputs.
   - providers/: Configurations for various AI model providers (OpenAI, Anthropic, Groq, Ollama).
8. Utilities (utils/):
   - computeSimilarity.ts: Calculates similarity between vectors.
   - formatHistory.ts: Formats chat history into strings.
   - logger.ts: Sets up application-wide logging using Winston.

The application is designed as a flexible and modular AI-powered chat system with web search capabilities, real-time communication, and persistent storage. It leverages modern libraries and frameworks for efficient development and scalability.
</result>
</attempt_completion>
