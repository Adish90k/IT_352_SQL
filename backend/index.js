const express = require('express');
const cors = require("cors");
const { exec } = require('child_process');
const app = express();
const port = 8000;

app.use(express.json());  

app.get("/",(req,res)=>{
    res.json({message:"ok"})
})

app.use(cors());
app.post('/api/calculate', (req, res) => {
    const { dpkts, doctets, srcaddr, dstaddr, input, output, srcport, dstport, prot, tos, tcp_flags } = req.body; 
    console.log(req.body); 
    // Construct the command to execute the Python script with the received data
    const command = `echo ${dpkts},${doctets},${srcaddr},${dstaddr},${input},${output},${srcport},${dstport},${prot},${tos},${tcp_flags} | python C:/Users/Adish/Documents/assignments/IT352/backend/knnCode.py`;

    // Execute the command
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${error.message}`);
            res.status(500).send('Internal server error');
            return;
        }
        if (stderr) {
            console.error(`stderr: ${stderr}`);
            res.status(500).send('Internal server error');
            return;
        }

        // Send the response
        res.send(stdout);
    });
});

app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
});
