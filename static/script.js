document.addEventListener("DOMContentLoaded", () => {
    const socket = io.connect();
    const startTalkingButton = document.getElementById("start-talking-button");
    const keyboardButton = document.getElementById("keyboard-button");
    const sendButton = document.getElementById("send-button");
    const stopAudioButton = document.getElementById("stop-audio-button");
    const userInput = document.getElementById("chat-input");
    const messagesDiv = document.getElementById("chat-messages");
    const hiddenInputContainer = document.getElementById("hidden-input-container");
    let loadingMessageElement = null;
    let loadingInterval = null;

    let mediaRecorder;
    let audioChunks = [];
    let stream;  // To store the media stream
    const audioPlayers = [];  // Array to keep track of all audio players

    hiddenInputContainer.style.display = 'none';

    // Handle incoming message and audio data from the server
    socket.on("response_with_audio", data => {
        removeLoadingIndicator();
        appendMessage("bot", data.message, data.audio);
    });

    // Handle STT response
    socket.on("stt_response", data => {
        removeLoadingIndicator();

        if (data.text) {
            appendMessage("user", data.text);
            showLoadingIndicator();
            socket.emit("message", data.text);
        } else {
            appendMessage("user", "Unable to detect human voice.");
            console.error('Failed to convert speech to text:', data.error);
        }
    });

    // Pause all audio players when stopAudioButton is clicked
    stopAudioButton.addEventListener("click", () => {
        audioPlayers.forEach(player => {
            player.pause();
            player.currentTime = 0;  // Reset audio to the start
        });
        console.log("All audio playback stopped.");
    });

    keyboardButton.addEventListener("click", () => {
        const isVisible = hiddenInputContainer.style.display === 'flex';
        hiddenInputContainer.style.display = isVisible ? 'none' : 'flex';
        hiddenInputContainer.style.flexDirection = 'row';

        if (!isVisible) {
            userInput.focus();
        }
    });

    sendButton.addEventListener("click", () => {
        const message = userInput.value.trim();
        if (message) {
            appendMessage("user", message);
            socket.emit("message", message);
            userInput.value = "";
            showLoadingIndicator();
        }
    });

    userInput.addEventListener("keyup", (event) => {
        if (event.key === "Enter" || event.keyCode === 13) {
            sendButton.click();
        }
    });

    startTalkingButton.addEventListener("click", () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            stopRecording();
            startTalkingButton.classList.remove("recording");
            startTalkingButton.textContent = "🎤";
        } else {
            startRecording();
            startTalkingButton.classList.add("recording");
            startTalkingButton.textContent = "🎤";
        }
    });

    function appendMessage(sender, message, audioData = null) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", sender.toLowerCase());
        messageElement.innerHTML = message;

        if (audioData) {
            // Generate a unique Blob URL for each audio message
            const audioBlob = new Blob([new Uint8Array(audioData)], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);

            // Create a new audio player instance for this message's audio and store it in the array
            const audioPlayer = new Audio(audioUrl);
            audioPlayers.push(audioPlayer);  // Track this audio player

            // Attempt to autoplay the audio
            audioPlayer.play().catch(error => {
                console.log("Autoplay failed: ", error);
            });

            // Create a replay button for this specific message
            const replayButton = document.createElement("button");
            replayButton.innerText = "▶️";  // Triangle play icon
            replayButton.classList.add("replay-button");

            // Replay functionality: reset audio and play from the beginning
            replayButton.addEventListener("click", () => {
                audioPlayer.currentTime = 0;  // Reset audio to the start
                audioPlayer.play().catch(err => {
                    console.error("Replay failed:", err);
                });
            });

            // Add replay button to the message element
            messageElement.appendChild(replayButton);
        }

        messagesDiv.appendChild(messageElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(userStream => {
                console.log("Microphone access granted");
                stream = userStream;  // Store the stream to release it later
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                console.log("MediaRecorder started");

                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                    console.log("Audio chunk recorded, size:", event.data.size);
                };

                mediaRecorder.onstop = () => {
                    console.log("MediaRecorder stopped");
                    if (audioChunks.length > 0) {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        console.log("Audio blob created, size:", audioBlob.size);
                        
                        const reader = new FileReader();
                        reader.onload = () => {
                            const arrayBuffer = reader.result;
                            const audioData = new Uint8Array(arrayBuffer);
                            console.log("Audio data converted to Uint8Array, length:", audioData.length);

                            socket.emit("speech_to_text", { audio: audioData });
                            console.log("Audio data sent to server for STT");
                        };

                        reader.onerror = (error) => {
                            console.error("Error reading audio blob:", error);
                        };

                        reader.readAsArrayBuffer(audioBlob);
                        showLoadingIndicator(true);
                    } else {
                        console.log("No audio data recorded.");
                        alert("No audio data recorded. Please try again.");
                    }

                    // Release the microphone after recording is finished
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                        stream = null;
                        mediaRecorder = null;  // Clear mediaRecorder for a fresh start
                    }
                };
            })
            .catch(error => {
                console.error("Error accessing microphone:", error);
                alert("Microphone access denied or unavailable. Please check your settings.");
            });
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
    }

    function showLoadingIndicator() {
        if (!loadingMessageElement) {
            loadingMessageElement = document.createElement("div");
            loadingMessageElement.classList.add("message", "loading");
            loadingMessageElement.textContent = ".";
            loadingMessageElement.style.textAlign = 'center';
            loadingMessageElement.style.alignSelf = 'center';

            messagesDiv.appendChild(loadingMessageElement);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            let dots = 1;
            loadingInterval = setInterval(() => {
                dots = (dots % 3) + 1;
                loadingMessageElement.textContent = ".".repeat(dots);
            }, 500);
        }
    }

    function removeLoadingIndicator() {
        if (loadingMessageElement) {
            messagesDiv.removeChild(loadingMessageElement);
            loadingMessageElement = null;
        }
        if (loadingInterval) {
            clearInterval(loadingInterval);
            loadingInterval = null;
        }
    }
});
