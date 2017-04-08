/* Dependencies */
import React from 'react';
import ReactDOM from 'react-dom';
import $ from 'jquery';

$(function() {
    var socket = io();
    $('form').submit(function() {
        socket.emit('chat message', $('#m').val());
        $('#m').val('');
        return false;
    });
    socket.on('chat message', function(msg) {
        $('#messages').append($('<li>').text(msg));
    });
});

ReactDOM.render(
    <div> HIHIHIHIHI</div>,
    document.getElementById("root")
);