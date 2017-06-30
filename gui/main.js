$('#send-lyrics').click(function(){
    var data = {text: $('#lyrics').val()};
    MIDIjs.play('?' + $.param(data));
});
