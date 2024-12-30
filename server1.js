const express = require("express");
const bodyParser = require("body-parser");
const twilio = require("twilio");

const app = express();
const PORT = 3000;

// Twilio Credentials
const accountSid = "your_account_sid"; // Replace with your Twilio Account SID
const authToken = "your_auth_token";   // Replace with your Twilio Auth Token
const client = twilio(accountSid, authToken);

// Store OTPs in memory (use a database for production)
let otpStorage = {};

app.use(bodyParser.json());
app.use(express.static("public"));

app.post("/send-otp", (req, res) => {
  const { phoneNumber } = req.body;
  if (!phoneNumber) {
    return res.status(400).json({ success: false, message: "Phone number is required" });
  }

  // Generate a 6-digit OTP
  const otp = Math.floor(100000 + Math.random() * 900000).toString();

  // Send OTP via Twilio
  client.messages
    .create({
      body: `Your OTP is ${otp}`,
      from: "+your_twilio_phone_number", // Replace with your Twilio phone number
      to: phoneNumber,
    })
    .then(() => {
      otpStorage[phoneNumber] = otp; // Store OTP
      res.json({ success: true, message: "OTP sent successfully!" });
    })
    .catch((err) => {
      console.error(err);
      res.status(500).json({ success: false, message: "Failed to send OTP" });
    });
});

app.post("/verify-otp", (req, res) => {
  const { phoneNumber, otp } = req.body;
  if (!otp || !phoneNumber) {
    return res.status(400).json({ success: false, message: "Phone number and OTP are required" });
  }

  if (otpStorage[phoneNumber] === otp) {
    delete otpStorage[phoneNumber]; // Clear OTP after verification
    res.json({ success: true, message: "OTP verified successfully!" });
  } else {
    res.status(400).json({ success: false, message: "Invalid OTP" });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
