var express = require('express');
var path = require('path');
var bodyParser = require('body-parser');

var Pusher = require('pusher');

var pusher = new Pusher({
  appId: "881937",
  key: "6d78a894c1acafc611d2",
  secret: "4e55702351a1988cb345",
  cluster: "us3",
  encrypted: true
});

var app = express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(express.static(path.join(__dirname, 'public')));

// Error Handler for 404 Pages
app.use(function(req, res, next) {
    var error404 = new Error('Route Not Found');
    error404.status = 404;
    next(error404);
});

module.exports = app;

app.listen(9000, function(){
  console.log('Example app listening on port 9000!')
});

app.get('/addTemperature', function(req,res){
  var temp = parseInt(req.query.temperature);
  var time = parseInt(req.query.time);
  if(temp && time && !isNaN(temp) && !isNaN(time)){
    var newDataPoint = {
      temperature: temp,
      time: time
    };
    londonTempData.dataPoints.push(newDataPoint);
    pusher.trigger('london-temp-chart', 'new-temperature', {
      dataPoint: newDataPoint
    });
    res.send({success:true});
  }else{
    res.send({success:false, errorMessage: 'Invalid Query Paramaters, required - temperature & time.'});
  }
});