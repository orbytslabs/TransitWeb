$(document).ready(function (e) {
    $('#finderImage').click(function (e) {
        var posX = $(this).offset().left,
            posY =  $(this).height() + $(this).offset().top;
        var sourcePosX = e.pageX - posX,
            sourcePosY = posY - e.pageY;
        alert(sourcePosX + ' , ' + sourcePosY);

        $.ajax({
          type: "POST",
          url: "/source-selection/create-photometry-source",
          data: { 
            "sourcePosX": sourcePosX, 
            "sourcePosY": sourcePosY,
            csrfmiddlewaretoken: $("input[name='csrfmiddlewaretoken']").val() 
          },
          success: function(response){
            console.log('Success');
          },
        });
    });
});