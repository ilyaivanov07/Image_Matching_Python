import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

fromaddr = "mikskuntz@gmail.com"
msg_subject = "OMSCS Survey"
msg_body = "Hello\n\nI am doing research for the OMSCS EdTech class. Please be kind to answer a couple of questions in this survey: \n\n https://docs.google.com/forms/d/e/1FAIpQLSf5YSiVYvwLdVMYMoqc8mcFEaMsq5g4cpq5wx6URsJsl0dWAQ/viewform?usp=sf_link"
server = smtplib.SMTP('doittsmtp.nycnet', 25)
server.starttls()

file_to_write = open("sent_emails","a")

with open("addresses","r+") as f:
    for line in f:
        toaddr = line.strip()
        msg = MIMEMultipart()
        msg['From'] = fromaddr
        msg['To'] = toaddr
        msg['Subject'] = msg_subject
        msg.attach(MIMEText(msg_body, 'plain'))
        text = msg.as_string()
        server.sendmail(fromaddr, toaddr, text)

        file_to_write.write(line)
        print(line)

file_to_write.write("\n")


# for i in range(1):
    #fromaddr = "iivanov@doitt.nyc.com"
    #toaddr = "iivanov8@gatech.edu"
    #toaddr = "mikskuntz@gmail.com"
    # toaddr = "iivanov67@yahoo.com"
    # msg = MIMEMultipart()
    # msg['From'] = fromaddr
    # msg['To'] = toaddr
    # msg['Subject'] = "Survey"
    # body = "Take the survey:\n\n https://docs.google.com/forms/d/e/1FAIpQLSf5YSiVYvwLdVMYMoqc8mcFEaMsq5g4cpq5wx6URsJsl0dWAQ/viewform?usp=sf_link"
    # msg.attach(MIMEText(body, 'plain'))
    # text = msg.as_string()
    # server.sendmail(fromaddr, toaddr, text)

server.quit()
file_to_write.close()