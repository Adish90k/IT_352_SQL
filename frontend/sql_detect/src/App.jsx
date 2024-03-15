import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const InputForm = () => {
  const [formData, setFormData] = useState({
    dpkts: "",
    doctets: "",
    srcaddr: "",
    dstaddr: "",
    input: "",
    output: "",
    srcport: "",
    dstport: "",
    prot: "",
    tos: "",
    tcp_flags: "",
  });
  const [sqloutput,setSqloutput] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      console.log(formData);
      const response = await axios.post("http://localhost:8000/api/calculate", formData);
      console.log(response.data);
      const outputstring = response.data;
      const valueWithinBrackets = outputstring.match(/\[(.*?)\]/)[1];
     
 
   if(valueWithinBrackets==0){
     setSqloutput("FALSE");
   }else if(valueWithinBrackets==1){
    setSqloutput("TRUE");
   }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div>
      <div className="namescontainer">
        <h1>IT352 PROJECT</h1>
        <h3>ADISH 211IT004</h3>
        <h3>MANJUNATH 211IT039</h3>
        <h3>KARTIK 211IT029</h3>
      </div>
      <div className="mainformcontainer">
        <h2>Enter Input</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            name="dpkts"
            value={formData.dpkts}
            onChange={handleChange}
            placeholder="dpkts"
          />
          <input
            type="text"
            name="doctets"
            value={formData.doctets}
            onChange={handleChange}
            placeholder="doctets"
          />
          <input
            type="text"
            name="srcaddr"
            value={formData.srcaddr}
            onChange={handleChange}
            placeholder="srcaddr"
          />
          <input
            type="text"
            name="dstaddr"
            value={formData.dstaddr}
            onChange={handleChange}
            placeholder="dstaddr"
          />
          <input
            type="text"
            name="input"
            value={formData.input}
            onChange={handleChange}
            placeholder="input"
          />
          <input
            type="text"
            name="output"
            value={formData.output}
            onChange={handleChange}
            placeholder="output"
          />
          <input
            type="text"
            name="srcport"
            value={formData.srcport}
            onChange={handleChange}
            placeholder="srcport"
          />
          <input
            type="text"
            name="dstport"
            value={formData.dstport}
            onChange={handleChange}
            placeholder="dstport"
          />
          <input
            type="text"
            name="prot"
            value={formData.prot}
            onChange={handleChange}
            placeholder="prot"
          />
          <input
            type="text"
            name="tos"
            value={formData.tos}
            onChange={handleChange}
            placeholder="tos"
          />
          <input
            type="text"
            name="tcp_flags"
            value={formData.tcp_flags}
            onChange={handleChange}
            placeholder="tcp_flags"
          />
        </form>
        <button type="submit" onClick={handleSubmit}>Submit</button>
      </div>
      <div className="ouputcontainer">
        <h2>Output:</h2>
        <span>{sqloutput}</span>
      </div>
    </div>
  );
};

export default InputForm;
