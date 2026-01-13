// server.js - Optimized PDF Chatbot with RAG
// Install: npm install express cors multer pdf-parse @anthropic-ai/sdk @xenova/transformers dotenv

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const Anthropic = require('@anthropic-ai/sdk');

const app = express();
const port = process.env.PORT || 3000;

// Increase payload limits
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const upload = multer({ storage: multer.memoryStorage() });

const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY
});

// Configure CORS to allow your frontend
const corsOptions = {
    origin: process.env.FRONTEND_URL || '*',
    methods: ['GET', 'POST'],
    credentials: true
};

app.use(cors(corsOptions));

// Store embeddings in memory (in production, use a vector DB)
const documentStore = new Map();

// Simple chunking function - splits by paragraphs
function chunkText(text, chunkSize = 1000, overlap = 200) {
    const chunks = [];
    const paragraphs = text.split(/\n\n+/);
    
    let currentChunk = '';
    let chunkIndex = 0;
    
    for (const para of paragraphs) {
        if ((currentChunk + para).length > chunkSize && currentChunk.length > 0) {
            chunks.push({
                id: chunkIndex++,
                text: currentChunk.trim()
            });
            
            // Keep overlap
            const words = currentChunk.split(' ');
            const overlapWords = words.slice(-Math.floor(overlap / 5));
            currentChunk = overlapWords.join(' ') + ' ' + para;
        } else {
            currentChunk += (currentChunk ? '\n\n' : '') + para;
        }
    }
    
    if (currentChunk.trim()) {
        chunks.push({
            id: chunkIndex++,
            text: currentChunk.trim()
        });
    }
    
    return chunks;
}

// Simple cosine similarity
function cosineSimilarity(a, b) {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magA * magB);
}

// Simple embedding using word frequency (TF-IDF inspired)
function simpleEmbedding(text, vocabulary = null) {
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    
    if (!vocabulary) {
        vocabulary = [...new Set(words)];
    }
    
    const vector = new Array(vocabulary.length).fill(0);
    const wordCounts = {};
    
    words.forEach(word => {
        wordCounts[word] = (wordCounts[word] || 0) + 1;
    });
    
    vocabulary.forEach((word, idx) => {
        vector[idx] = wordCounts[word] || 0;
    });
    
    return { vector, vocabulary };
}

app.get('/', (req, res) => {
    res.json({ status: 'RAG-powered PDF Chatbot Server Running' });
});

app.post('/upload', upload.single('pdf'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                success: false, 
                error: 'No file uploaded' 
            });
        }

        console.log('Parsing PDF...');
        const pdfData = await pdfParse(req.file.buffer);
        const text = pdfData.text;

        console.log('Chunking document...');
        const chunks = chunkText(text, 800, 150);
        
        console.log(`Created ${chunks.length} chunks`);

        // Create vocabulary from all chunks
        const allWords = chunks.flatMap(chunk => 
            chunk.text.toLowerCase().match(/\b\w+\b/g) || []
        );
        const vocabulary = [...new Set(allWords)].slice(0, 500); // Limit vocabulary size

        // Create embeddings for each chunk
        console.log('Creating embeddings...');
        const chunksWithEmbeddings = chunks.map(chunk => {
            const { vector } = simpleEmbedding(chunk.text, vocabulary);
            return {
                ...chunk,
                embedding: vector
            };
        });

        // Generate unique document ID
        const docId = Date.now().toString();
        
        // Store in memory
        documentStore.set(docId, {
            chunks: chunksWithEmbeddings,
            vocabulary: vocabulary,
            pages: pdfData.numpages,
            totalChunks: chunks.length
        });

        console.log('Document processed and stored');

        res.json({ 
            success: true, 
            docId: docId,
            pages: pdfData.numpages,
            chunks: chunks.length,
            message: `Processed ${pdfData.numpages} pages into ${chunks.length} searchable chunks`
        });
    } catch (error) {
        console.error('Error processing PDF:', error);
        res.status(500).json({ 
            success: false, 
            error: 'Failed to process PDF: ' + error.message 
        });
    }
});

app.post('/ask', async (req, res) => {
    try {
        const { question, docId } = req.body;

        if (!question || !docId) {
            return res.status(400).json({ 
                success: false, 
                error: 'Missing question or document ID' 
            });
        }

        const doc = documentStore.get(docId);
        if (!doc) {
            return res.status(404).json({ 
                success: false, 
                error: 'Document not found. Please re-upload.' 
            });
        }

        console.log(`Question: ${question}`);

        // Create embedding for the question
        const { vector: questionVector } = simpleEmbedding(question, doc.vocabulary);

        // Find most relevant chunks
        const scoredChunks = doc.chunks.map(chunk => ({
            ...chunk,
            similarity: cosineSimilarity(questionVector, chunk.embedding)
        }));

        // Get top 3 most relevant chunks
        const topChunks = scoredChunks
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 3);

        console.log(`Top similarities: ${topChunks.map(c => c.similarity.toFixed(3)).join(', ')}`);

        // Combine relevant context
        const context = topChunks.map(c => c.text).join('\n\n---\n\n');

        // Create prompt with only relevant context
        const prompt = `You are a helpful, friendly assistant answering questions about a document. 

Here are the most relevant sections from the document:

<document>
${context}
</document>

Please answer the user's question in a natural, conversational way:
- Explain the information clearly and concisely
- Use bullet points only when listing multiple items makes it clearer
- Organize the information logically
- If there are specific requirements or rules, present them in an easy-to-understand format
- Be helpful and direct - avoid unnecessary preambles like "Based on the document..."
- If the answer isn't in these sections, politely say so and suggest what information might be needed

Question: ${question}

Answer:`;

        console.log(`Sending ${context.length} characters to Claude (reduced from full document)`);

        // Call Claude API
        const message = await anthropic.messages.create({
            model: 'claude-sonnet-4-20250514',
            max_tokens: 1024,
            messages: [{
                role: 'user',
                content: prompt
            }]
        });

        const answer = message.content[0].text;

        res.json({ 
            success: true, 
            answer: answer,
            chunksUsed: topChunks.length,
            contextSize: context.length
        });
    } catch (error) {
        console.error('Error calling Claude API:', error);
        res.status(500).json({ 
            success: false, 
            error: error.message || 'Failed to get answer from AI' 
        });
    }
});

app.listen(port, () => {
    console.log(`RAG-powered PDF Chatbot running on http://localhost:${port}`);
    console.log('API Key configured:', !!process.env.ANTHROPIC_API_KEY);
});