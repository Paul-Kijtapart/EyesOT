/* Dependencies */
import React from 'react';
import ReactDOM from 'react-dom';
import $ from 'jquery';
import io from 'socket.io-client';

let socket = io('http://localhost:3000');

// $(function() {
//     var socket = io();
//     $('form').submit(function() {
//         socket.emit('chat message', $('#m').val());
//         $('#m').val('');
//         return false;
//     });
//     socket.on('chat message', function(msg) {
//         $('#messages').append($('<li>').text(msg));
//     });
// });


class App extends React.Component {
    constructor(props) {
        super(props);
        this.onFormSubmit = this.onFormSubmit.bind(this);
    }

    componentDidMount() {
        socket.on("chat message", function(msg) {
            $('#messages').append($('<li>').text(msg));
        });
    }

    onFormSubmit(e) {
        e.preventDefault();
        socket.emit('chat message', $('#m').val());
        $('#m').val('');
        return false;
    }

    render() {
        return (
            <div className="app">
                <ul id="messages"></ul>
                <form onSubmit={this.onFormSubmit} action="">
                <input id="m" autocomplete="off" />
                    <button>Send</button>
                </form>
            </div>
        );
    }
}

ReactDOM.render(
    <App />,
    document.getElementById("root")
);