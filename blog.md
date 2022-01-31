---
layout: page
title: ""
---



<ul>
{% for post in site.posts %}
    {% if post.categories contains "blog" %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
{% endfor %}
</ul>