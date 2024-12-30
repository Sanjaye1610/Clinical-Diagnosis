const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
    res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type');
    next();
});

app.post('/chat', async (req, res) => {
    const userMessage = req.body.message;
    
    const apiKey = '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';
    
    const apiUrl = 'https://api.openai.com/v1/chat/completions'; 
    try {
        const response = await axios.post(apiUrl, {
            model: 'gpt-3.5-turbo', 
            messages: [{ role: 'user', content: userMessage }],
            max_tokens: 150,
        }, {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            },
        });

        const botReply = response.data.choices[0].message.content.trim(); 
        res.json({ reply: botReply });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ reply: 'Sorry, something went wrong.' });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
