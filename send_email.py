import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# server = smtplib.SMTP('doittsmtp.nycnet', 25)
# msg = "YOUR MESSAGE!"
# server.sendmail("iivanov@doitt.nyc.gov", "mikskuntz@gmail.com", msg)
# server.quit()


server = smtplib.SMTP('doittsmtp.nycnet', 25)
server.starttls()

for i in range(1):
    # fromaddr = "iivanov@doitt.nyc.com"
    fromaddr = "iivanov67@gmail.com"
    toaddr = "iivanov67@yahoo.com"
    #toaddr = "mikskuntz@gmail.com"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Survey"
    body = "Take the survey:\n\n https://docs.google.com/forms/d/e/1FAIpQLSf5YSiVYvwLdVMYMoqc8mcFEaMsq5g4cpq5wx6URsJsl0dWAQ/viewform?usp=sf_link"
    msg.attach(MIMEText(body, 'plain'))
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)

server.quit()