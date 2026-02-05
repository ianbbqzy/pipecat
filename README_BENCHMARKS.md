git clone https://YOUR_TOKEN@github.com/ianbbqzy/pipecat.git

cd pipecat

# Setup venv
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e '.[inworld,elevenlabs,cartesia]'
pip install python-dotenv aiohttp

# Set your API keys
export INWORLD_API_KEY=your_key_here

# Run benchmark (just Inworld to compare)
python benchmark_word_tts_ttfb.py --services inworld
