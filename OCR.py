import cv2
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# ==========================================
# 1. INITIALIZE GEMINI AI SECURELY
# ==========================================
load_dotenv() # Loads the hidden variables from the .env file
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: API key not found. Please check your .env file.")
    exit()

client = genai.Client(api_key=api_key) 

# ==========================================
# 2. VALIDATION DATABASE
# ==========================================
WESTERN_LINE = ["CHURCHGATE", "MARINE LINES", "CHARNI ROAD", "GRANT ROAD", "MUMBAI CENTRAL", "MAHALAXMI", "LOWER PAREL", "PRABHADEVI", "DADAR", "MATUNGA ROAD", "MAHIM", "BANDRA", "KHAR ROAD", "SANTACRUZ", "VILE PARLE", "ANDHERI", "JOGESHWARI", "RAM MANDIR", "GOREGAON", "MALAD", "KANDIVALI", "BORIVALI", "DAHISAR", "MIRA ROAD", "BHAYANDAR", "NAIGAON", "VASAI ROAD", "NALASOPARA", "VIRAR"]
CENTRAL_LINE = ["CSMT", "MASJID", "SANDHURST ROAD", "BYCULLA", "CHINCHPOKLI", "CURREY ROAD", "PAREL", "DADAR", "MATUNGA", "SION", "KURLA", "GHATKOPAR", "THANE", "DOMBIVLI", "KALYAN", "ULHASNAGAR", "AMBERNATH"]
HARBOUR_LINE = ["MAHIM", "BANDRA", "KHAR ROAD", "SANTACRUZ", "VILE PARLE", "ANDHERI", "JOGESHWARI", "RAM MANDIR", "GOREGAON","KINGS CIRCLE", "VADALA ROAD", "SEWRI", "CHEMBUR", "VASHI", "NERUL", "BELAPUR", "PANVEL"]
# Combine them into one string for the AI
ALL_STATIONS = ", ".join(set(CENTRAL_LINE + HARBOUR_LINE))


def extract_ticket_data(image_frame):
    print("\n" + "="*40)
    print("🧠 SENDING TICKET TO GEMINI 2.5 FLASH...")
    print("="*40)
    
    # 1. Boost contrast and brightness to separate faded ink from dark borders
    alpha = 1.5  # Contrast control (1.5 is a 50% boost. Increase to 2.0 if still failing)
    beta = 20    # Brightness control (0 is normal. Higher makes the background whiter)
    enhanced_frame = cv2.convertScaleAbs(image_frame, alpha=alpha, beta=beta)

    # 2. Convert the ENHANCED frame to standard RGB for the AI
    rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # 3. Dynamic Prompt Injection (Notice the 'f' string format)
    prompt = f"""
    You are an expert at reading Mumbai Local train tickets. You must be able to read BOTH physically printed dot-matrix tickets and digital 'Railone' app tickets.
    
    First, analyze the image to determine the Ticket Category: "Journey Ticket", "Return Ticket", or "Season Pass".
    Use context to fix blurry or faded text on physical tickets.

    CRITICAL SPELLING RULE: 
    The Mumbai rail network only contains specific stations. The station names you extract MUST perfectly match a station from this allowed list: 
    [{ALL_STATIONS}]
    
    If the ticket text is blurry and looks like a typo (e.g., "BADAR", "BAJAR", or "DADR"), you MUST map it to the closest valid station from the list above (e.g., "DADAR"). Do not output invalid station names.

    Extract the following data strictly in JSON format:
    
    COMMON FIELDS (Must always be extracted):
    - "Ticket ID / UTS No.": Look explicitly for the text "UTS:" or "UTS No:". The ID is the alphanumeric string immediately following it. CRITICAL: Do NOT read the number next to the letter 'M' in the top corners.
    - "Source Station": Extract ONLY the alphabetical station name. You MUST strip out any distance numbers, brackets, or kilometer markers (e.g., if you see "DADAR(27)" or "DADAR -27 km-", output strictly "DADAR"). MUST be from the allowed list.
    - "Destination Station": Extract ONLY the alphabetical station name. You MUST strip out any ampersands ("&") and distance markers (e.g., if you see "& KALYAN", output strictly "KALYAN"). MUST be from the allowed list.
    - "Ticket Class": Must be exactly "First Class", "Second Class", or "AC EMU".
    - "Ticket Category": Must be "Journey Ticket", "Return Ticket", or "Season Pass".

    CONDITIONAL DATE FIELDS (Follow these rules strictly based on the Ticket Category):
    - If "Journey Ticket" or "Return Ticket": 
        Extract "Booking Date & Time" (Format DD/MM/YYYY HH:MM). 
        CRITICAL HINT: On physical tickets, this is usually printed in faded dot-matrix ink near the very top or bottom edge. It heavily overlaps with the pre-printed borders. Look extremely closely at the border lines for hidden digits. If a number is cut in half by a line, use the visible half to infer the digit.
        Set "Valid From Date" and "Valid To Date" to "Not Applicable".
    - If "Season Pass": 
        Extract "Valid From Date" and "Valid To Date" (Format DD-MM-YYYY). 
        Set "Booking Date & Time" to "Not Applicable".

    If any required field is completely unreadable due to glare or damage, output the value as "Not Found".
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        data = json.loads(response.text)

        print("\n✨ TICKET DATA SUCCESSFULLY EXTRACTED")
        print("-" * 40)
        for key, value in data.items():
            print(f"{key.ljust(22)}: {value}")
        print("-" * 40 + "\n")

    except Exception as e:
        print(f"\n❌ Network or API Error: Could not verify ticket.")
        print(f"Details: {e}")
        print("Please check your internet connection and try again.\n")


def start_live_scanner():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("\n" + "="*50)
    print("📷 CLOUD AI TICKET SCANNER ACTIVATED")
    print("-> Hold the ticket up to your camera.")
    print("-> Press the 'SPACE' bar to snap a photo.")
    print("-> The scanner will process the image and close automatically.")
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
            print("Shutting down scanner camera...")
            break
            
        elif key == ord('q'):
            print("\nClosing scanner. Goodbye!")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_scanner()