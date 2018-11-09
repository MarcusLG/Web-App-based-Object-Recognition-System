var openFile = function(file) {
	var input = file.target;
	var reader = new FileReader();
	reader.onload = function(){
		var dataURL = reader.result;
		var output = document.getElementById('output');
		output.src = dataURL;
	};
	reader.readAsDataURL(input.files[0]);
};
async function model_loading(tensor){
	const model= await tf.loadModel('http://192.168.1.29:8080/model.json');
	document.getElementById("Line_2").innerHTML = "Processing..";
	console.log("Model loaded button");
	console.log(model);
	let prediction = await model.predict(tensor);
	prediction.print();
    list=["\tD13","\tD24","\tD175","\tD197"];
    output_data=prediction.dataSync();
    var x=0;
    var mx=output_data[0];
    if (output_data[1]>mx){
    	x=1;
    	mx=output_data[1];
    }
    if (output_data[2]>mx){
    	x=2;
    	mx=output_data[2]
    }
    if (output_data[3]>mx) {
    	x=3;
    }
    document.getElementById("Line_2").innerHTML = " ";  
    document.getElementById("Line_3").innerHTML = list[x];
 
}
$("#predict-button").click(async function(){
    //Initialize the image object
    var image=document.getElementById('output');
    document.getElementById("Line_2").innerHTML = "Processing.";
    console.log(image);
    //convert the image object to a tensor by resizing it and Normalizing it using the ImageNet mean RGB values
    let tensor = tf.fromPixels(image,1).resizeNearestNeighbor([200,200]).toFloat();
    console.log(tensor);
    tensor=tensor.reshape([1,200,200,1]);
    console.log(tensor);
    model_loading(tensor);
	//	model_loading();
    //define the Prediction object and put a future event for prediction.
	//let prediction = await model.predict(tensor);
	//prediction.print();
});