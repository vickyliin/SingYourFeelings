var host = "54.254.210.178",
    port = "5678";
var ws = new WebSocket("ws://" + host + ":" + port);
ws.onmessage = function (event) {
  MIDIjs.play("/midifiles/" + event.data);
};
$('#send-lyrics').click(function(){
  ws.send($('#lyrics').val());
});
