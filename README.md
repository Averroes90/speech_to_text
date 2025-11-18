# Multi-Service Video Transcription & Translation Pipeline

A robust, scalable backend system for automated video transcription and translation using multiple AI services. The architecture employs factory and adapter design patterns to provide a unified interface across different transcription providers, currently supporting Google Cloud Speech-to-Text (Chirp models) and OpenAI Whisper.

## Overview

This project provides an enterprise-grade solution for extracting, transcribing, and translating audio from video files. It processes videos concurrently using multiple transcription services, enabling quality comparison and future ensemble methods for improved accuracy.

### Key Features

- **Multi-Service Architecture**: Concurrent processing through Google Cloud and OpenAI APIs
- **Factory Pattern Implementation**: Extensible design for easy integration of new transcription services
- **Automatic Audio Extraction**: Handles various video formats (MP4, WMV, MOV, MPG)
- **Parallel Processing**: ThreadPoolExecutor-based concurrent transcription for improved performance
- **SRT Output**: Industry-standard subtitle format generation
- **Language Support**: Extensive language coverage through both Google and OpenAI services

## Architecture

### Design Patterns

The system implements several design patterns for maintainability and extensibility:

- **Factory Pattern**: Service handlers are created through factory methods, allowing runtime service selection
- **Adapter Pattern**: Each service (Google, OpenAI) has dedicated adapters conforming to common protocols
- **Protocol-Based Design**: Abstract base classes define contracts for transcription, translation, and cloud services

### Core Components

```
├── handlers_and_protocols/
│   ├── protocols.py          # Abstract base classes defining service contracts
│   └── handlers.py           # Factory methods for service instantiation
├── adapters/
│   ├── google_adapters/      # Google Cloud service implementations
│   │   ├── google_transcribe_adapter.py
│   │   ├── google_translate_adapter.py
│   │   ├── GCP_adapter.py    # Cloud storage operations
│   │   └── google_environment_loader.py
│   └── openai_adapters/      # OpenAI service implementations
│       ├── openai_transcribe_adapter.py
│       └── openai_environment_loader.py
├── utils/
│   ├── audio_utils.py        # Audio extraction and processing
│   ├── nlp_utils2.py         # Response processing and NLP operations
│   └── utils.py              # General utilities
├── app.py                    # Main application logic and pipelines
└── main_concurrent_Multi.ipynb  # Development interface
```

## Service Workflows

### Google Cloud Pipeline

1. **Audio Extraction**: Extract audio from video file (FLAC format for optimal quality)
2. **Cloud Upload**: Upload audio to Google Cloud Storage bucket
3. **Transcription**: Utilize Chirp or Chirp-2 models via long-audio endpoint
4. **Translation**: Process transcribed text through Google Translate API
5. **SRT Generation**: Format timestamps and text into SRT format
6. **Cleanup**: Delete temporary audio files from cloud storage

### OpenAI Whisper Pipeline

1. **Audio Extraction**: Extract and segment audio into chunks (< Whisper threshold)
2. **Parallel Processing**: Concurrent transcription and translation of chunks
3. **Chunk Assembly**: Reconstruct complete transcript from processed segments
4. **SRT Generation**: Create timestamped subtitle file

## Installation

### Prerequisites

- Python 3.8+
- Google Cloud Platform account with Speech-to-Text and Translation APIs enabled
- OpenAI API key
- ffmpeg for audio extraction

### Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd transcription-pipeline
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage

### Jupyter Notebook Interface (Development)

The project includes a Jupyter notebook for interactive development and testing:

```python
import app
import utils.utils as utils

# Configure languages
source_language = "it"  # Italian
target_language = "en"  # English

# Process single video with multiple services
responses = app.multi_transcribe(
    file_path="/path/to/video.mp4",
    service_names=["google", "openai"],
    source_language=source_language,
    target_language=target_language,
    server_region="us-central1"  # For Google Chirp
)
```

### Programmatic Usage

```python
from app import transcribe_and_translate
import io

# Process audio data
with open("audio.flac", "rb") as f:
    audio_data = io.BytesIO(f.read())

srt_output = transcribe_and_translate(
    audio_data=audio_data,
    service_name="google",
    source_language="it",
    target_language="en",
    env_loaded=True,
    server_region="us-central1"
)
```

## API Service Details

### Google Cloud Configuration

- **Models**: Chirp, Chirp-2 (latest generation speech recognition)
- **Regions**: Configurable (us-central1, europe-west4)
- **Storage**: Temporary audio storage in Google Cloud Storage
- **Limits**: Supports long-form audio through cloud storage

### OpenAI Configuration

- **Model**: Whisper (automatic model selection)
- **Processing**: Chunk-based with automatic segmentation
- **Features**: Combined transcription and translation
- **Limits**: Automatic handling of file size constraints

## Core Protocols

The system defines three main protocol interfaces:

1. **TranscribeServiceHandler**: Audio transcription operations
2. **TranslationServiceHandler**: Text translation operations  
3. **CloudServiceHandler**: Cloud storage operations
4. **EnvironmentHandler**: Environment and credential management

## Dependencies

Key dependencies include:
- `google-cloud-speech` (2.26.0) - Google Speech-to-Text
- `google-cloud-translate` (3.15.3) - Google Translation
- `google-cloud-storage` (2.16.0) - Cloud storage operations
- `openai` (1.26.0) - OpenAI Whisper API
- `pydub` (0.25.1) - Audio processing
- `srt` (3.5.3) - SRT file generation
- `concurrent.futures` - Parallel processing

See `requirements.txt` for complete dependency list.

## Performance Considerations

- **Concurrent Processing**: Both services process simultaneously, reducing total wait time
- **Thread Safety**: Unique file naming prevents conflicts in multi-threaded operations
- **Memory Management**: Automatic garbage collection after large audio processing
- **Error Recovery**: Comprehensive error handling with automatic cleanup

## Future Development

### Planned Enhancements

1. **Ensemble Methods**: Combine outputs from multiple services for improved accuracy
2. **Additional Services**: Integration with AWS Transcribe, Azure Speech, AssemblyAI
3. **Web Interface**: Migration from Jupyter notebook to production web application
4. **Cloud Deployment**: Containerization and cloud-native deployment
5. **Advanced NLP**: Internal prompting techniques for output optimization
6. **Real-time Processing**: Support for streaming transcription

### Extension Points

The factory pattern allows easy addition of new services:

1. Implement service-specific adapters conforming to protocol interfaces
2. Register new service in factory methods
3. Add environment configuration handler
4. Service automatically available throughout the pipeline

## Development Status

Currently in active development with production deployment planned. The backend architecture is mature and tested, with frontend development and cloud deployment infrastructure in progress.

## Contributing

The project follows standard Python development practices:
- Type hints for improved code clarity
- Abstract base classes for protocol definition
- Comprehensive error handling
- Thread-safe operations for concurrent processing

## License

[Specify License]

## Contact

[Contact Information]

---

*This project represents a sophisticated approach to multi-service transcription, designed for scalability, accuracy comparison, and future ensemble methods. The architecture supports both immediate production use and long-term extensibility.*