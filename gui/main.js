var host = "54.254.210.178",
    port = "5678";

var ws = null;
onmessage = function(event){
  MIDIjs.play("/midifiles/" + event.data);
};
start = function(){
  ws = new WebSocket("ws://" + host + ":" + port);
  ws.onmessage = onmessage;
  ws.onclose = start;
};

start();
$('#send-lyrics').click(function(){
  ws.send($('#lyrics').val());
});
