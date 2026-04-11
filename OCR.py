import cv2
import json
from google import genai
from google.genai import types
from PIL import Image

# ==========================================
# 1. INITIALIZE GEMINI AI
# ==========================================
# PASTE YOUR API KEY BETWEEN THE QUOTES BELOW:
client = genai.Client(api_key="AIzaSyCSi30glIVelbWicXk2j2zT8SUpg2nqnlE") 

def extract_ticket_data(image_frame):
    print("\n" + "="*40)
    print("🧠 SENDING TICKET TO GEMINI AI...")
    print("="*40)
    
    # Convert OpenCV's BGR color format to standard RGB for the AI
    rgb_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # The "Smart Brain" prompt containing your exact business logic
    prompt = """
    You are an expert at reading Mumbai Local train tickets. You must be able to read BOTH physically printed dot-matrix tickets and digital 'Railone' app tickets.
    
    First, analyze the image to determine the Ticket Category: "Journey Ticket", "Return Ticket", or "Season Pass".
    Use context to fix blurry or faded text on physical tickets (e.g., "DADAR(27) To CRIIICH GTE" means Destination is "CHURCHGATE").

    Extract the following data strictly in JSON format:
    
    COMMON FIELDS (Must always be extracted):
    - "Ticket ID / UTS No.": The 10-character alphanumeric ID.
    - "Source Station": The starting station.
    - "Destination Station": The ending station.
    - "Ticket Class": Must be exactly "First Class", "Second Class", or "AC EMU".
    - "Ticket Category": Must be "Journey Ticket", "Return Ticket", or "Season Pass".

    CONDITIONAL DATE FIELDS (Follow these rules strictly based on the Ticket Category):
    - If "Journey Ticket" or "Return Ticket": 
        Extract "Booking Date & Time" (Format DD/MM/YYYY HH:MM). 
        Set "Valid From Date" and "Valid To Date" to "Not Applicable".
    - If "Season Pass": 
        Extract "Valid From Date" and "Valid To Date" (Format DD-MM-YYYY). 
        Set "Booking Date & Time" to "Not Applicable".

    If any required field is completely unreadable due to glare or damage, output the value as "Not Found".
    """

    try:
        # Send the image and our instructions to the gemini-1.5-flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        # Parse the JSON response back into a Python dictionary
        data = json.loads(response.text)

        # ==========================================
        # TERMINAL OUTPUT
        # ==========================================
        print("\n✨ GEMINI EXTRACTED TICKET DATA")
        print("-" * 40)
        for key, value in data.items():
            print(f"{key.ljust(22)}: {value}")
        print("-" * 40)
        print("Ready for next ticket. Press 'SPACE' to scan or 'Q' to quit.\n")

    # This catches network drops, API limit errors, or AI hallucinations gracefully
    except Exception as e:
        print(f"\n❌ Network or API Error: Could not verify ticket.")
        print(f"Details: {e}")
        print("Please check your internet connection and try again.")
        print("\nReady for next ticket. Press 'SPACE' to scan or 'Q' to quit.\n")


def start_live_scanner():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("\n" + "="*50)
    print("📷 AI LIVE TICKET SCANNER ACTIVATED")
    print("-> Hold the ticket up to your camera.")
    print("-> Press the 'SPACE' bar to snap a photo and scan.")
    print("-> Press 'Q' to close the scanner.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break
            
        cv2.imshow("Mumbai Local Ticket Scanner", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32: 
            extract_ticket_data(frame)
        elif key == ord('q'):
            print("\nClosing scanner. Goodbye!")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_scanner()